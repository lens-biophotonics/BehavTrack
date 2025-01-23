// Fetching references to DOM elements by their IDs
const folderInput = document.getElementById("folderInput"); // Button for selecting a folder containing images and annotations
const prevBtn = document.getElementById("prevBtn"); // Button for navigating to the previous image
const nextBtn = document.getElementById("nextBtn"); // Button for navigating to the next image
const annotationList = document.getElementById("annotationItems"); // Section displaying a list of annotations for the current image
const canvas = document.getElementById("annotationCanvas"); // Canvas element for displaying the image and drawing annotations
const ctx = canvas.getContext("2d"); // 2D drawing context for rendering on the canvas



// State management variables
let images = []; // Array to hold the list of images in the loaded folder
let currentImageIndex = 0; // Index of the currently displayed image in the `images` array
let annotations = {}; // Object to store annotations for each image, keyed by the image filename
let currentBBox = null; // Temporary storage for the bounding box being drawn
let dragging = false; // Boolean to track if a bounding box is currently being drawn (mousedown and drag event)
let mode = "bbox"; // Active mode: either "bbox" for drawing bounding boxes or "keypoint" for adding keypoints
let currentKeypointIndex = 0; // Tracks which keypoint (e.g., nose, earL) is being placed in "keypoint" mode
const keypoints = ["nose", "earL", "earR", "tailB"]; // Ordered array of keypoint names for consistent storage and display
let annotationFileHandle = null; // File handle for reading/writing the annotation.json file in the selected folder
let undoStack = []; // Stack to keep track of changes for undo functionality
let visibleKeypoint = false;
let selectedBBoxIndex = 0;

// Function to update the mode display in the toolbar
function updateModeDisplay() {
  // Dynamically update the text displayed in the mode indicator based on the current mode
  document.getElementById("modeDisplay").textContent =
    mode === "bbox" // If mode is "bbox", show "Bounding Box"
      ? "Mode: Bounding Box"
      // If mode is "keypoint", show the current keypoint being placed (e.g., "Keypoint (nose)")
      : `Mode: Keypoint (${keypoints[currentKeypointIndex]}) - Visibility (${visibleKeypoint})`;
}

function updateSelectedBBoxDisplay(index) {
  // Dynamically update the text displayed in the mode indicator based on the current mode
  document.getElementById("selectedBBox").textContent =
    `Selected Bounding Box: ${index+1}`;
}



function updateFileListAndProgress() {
  const fileList = document.getElementById("fileList"); // Get the file list container on the left-hand side
  fileList.innerHTML = ""; // Clear any existing entries in the file list to avoid duplication

  // Loop through all images in the folder
  images.forEach((image, index) => {
      const imageName = image.name; // Get the name of the current image
      const imageAnnotations = annotations[imageName] || []; // Retrieve annotations for the image, defaulting to an empty array if none exist

      // Check if the frame is fully annotated
      // A fully annotated frame requires exactly 5 bounding boxes, each with 4 keypoints
      const isComplete =
          imageAnnotations.length === 5 &&
          imageAnnotations.every(annotation => Object.keys(annotation.keypoints).length === 4);

      // Check if the frame is partially annotated
      // A partially annotated frame has at least one bounding box but does not meet the criteria for completeness
      const isPartial = imageAnnotations.length > 0 && !isComplete;

      // Create a new div element for this image in the file list
      const fileItem = document.createElement("div");
      fileItem.textContent = imageName; // Display the image name in the list
      fileItem.className = index === currentImageIndex ? "selected" : ""; // Highlight the current image being viewed

      // Assign a color to the file item based on its annotation completeness
      if (isComplete) {
          fileItem.style.color = "green"; // Green indicates the frame is fully annotated
      } else if (isPartial) {
          fileItem.style.color = "orange"; // Orange indicates the frame is partially annotated
      } else {
          fileItem.style.color = "black"; // Gray indicates the frame has no annotations
      }

      // Add a click event to allow the user to select this image
      fileItem.addEventListener("click", () => {
          currentImageIndex = index; // Update the current image index to this image
          loadImage(); // Load the selected image and its annotations onto the canvas
          updateFileListAndProgress(); // Refresh the file list to reflect the newly selected image
      });

      // Append the file item to the file list container
      fileList.appendChild(fileItem);
  });
}

  

// Event listener for the "folderInput" button to load a folder
folderInput.addEventListener("click", async () => {
  // Open the folder picker dialog and get a handle to the selected directory
  const directoryHandle = await window.showDirectoryPicker();

  // Initialize/reset the list of images and annotations
  images = []; // Array to store metadata about images in the folder
  annotations = {}; // Object to store annotations loaded from the annotation.json file

  // Iterate over the files in the selected directory
  for await (const entry of directoryHandle.values()) {
    // Check if the entry is an image file (based on its extension: .png or .jpg/.jpeg)
    if (entry.kind === "file" && /\.(png|jpe?g)$/i.test(entry.name)) {
      images.push({ name: entry.name, handle: entry }); // Add the image to the `images` array
    }

    // Check if the entry is the annotation.json file
    if (entry.name === "annotation.json") {
      annotationFileHandle = entry; // Store the handle for annotation.json for later reading/writing
    }
  }

  // Sort the `images` array alphabetically by image name to ensure consistent order
  images.sort((a, b) => a.name.localeCompare(b.name));

  // Check if the annotation.json file exists
  if (!annotationFileHandle) {
    // If annotation.json doesn't exist, create a new one
    annotationFileHandle = await directoryHandle.getFileHandle("annotation.json", {
      create: true, // Create a new file if it doesn't already exist
    });

    // Initialize the file with an empty annotations object
    saveAnnotations(); // Save the empty object to the new file
  } else {
    // If annotation.json exists, read its contents
    const file = await annotationFileHandle.getFile();
    annotations = JSON.parse(await file.text()); // Parse the JSON content into the `annotations` object
  }

  // Set the current image index to the first image in the folder
  currentImageIndex = 0;

  // Load the first image and its annotations onto the canvas
  loadImage();

  // Update the file list in the UI to reflect the loaded images and their annotation status
  updateFileListAndProgress();
});




// Function to load the currently selected image and its annotations onto the canvas
async function loadImage() {
  // Check if there are any images loaded; if not, exit the function
  if (images.length === 0) return;

  // Get the file handle for the currently selected image
  const imageFile = await images[currentImageIndex].handle.getFile();

  // Create a temporary URL for the image to be used in the <canvas> element
  const imageURL = URL.createObjectURL(imageFile);

  // Create a new HTMLImageElement to load the image
  const img = new Image();
  img.src = imageURL; // Set the source of the image element to the temporary URL
  img.onload = () => {
    // Once the image has loaded:
    // 1. Set the canvas dimensions to match the image dimensions
    canvas.width = img.width;
    canvas.height = img.height;

    // 2. Clear the canvas to remove any previously drawn content
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 3. Draw the image on the canvas at position (0, 0)
    ctx.drawImage(img, 0, 0);

    // 4. Draw existing annotations (bounding boxes and keypoints) for this image
    drawAnnotations();
  };

  // Update the annotation list displayed in the sidebar to reflect the annotations for the current image
  updateAnnotationList();
}



// Function to draw bounding boxes and keypoints on the canvas for the current image
function drawAnnotations() {
  // Get the name of the currently displayed image
  const imageName = images[currentImageIndex].name;

  // Retrieve the annotations for the current image, defaulting to an empty array if no annotations exist
  const imageAnnotations = annotations[imageName] || [];

  // Iterate over each annotation for the current image
  imageAnnotations.forEach(({ bbox, keypoints }, index) => {
    // --- Draw Bounding Box ---
    ctx.strokeStyle = "red"; // Set the stroke color for the bounding box to red
    ctx.lineWidth = 2; // Set the line width for the bounding box
    ctx.strokeRect(
      bbox.x1, // X-coordinate of the top-left corner
      bbox.y1, // Y-coordinate of the top-left corner
      bbox.x2 - bbox.x1, // Width of the bounding box
      bbox.y2 - bbox.y1 // Height of the bounding box
    );

    // --- Draw Keypoints ---
    for (const [key, coords] of Object.entries(keypoints)) {
      // `key` is the name of the keypoint (e.g., "nose", "earL")
      // `coords` is an array containing the X and Y coordinates of the keypoint

      if (coords[2] == 1) { // if the keypoint invisible 
        ctx.fillStyle = "gray"; // Set the fill color for the keypoint to yellow #8b930a
        ctx.beginPath(); // Begin a new path for the keypoint circle
        ctx.arc(
          coords[0], // X-coordinate of the keypoint
          coords[1], // Y-coordinate of the keypoint
          5, // Radius of the keypoint circle
          0, // Start angle (0 radians)
          2 * Math.PI // End angle (full circle)
        );
        ctx.fill(); // Fill the keypoint circle with the current fill color
  
        // Draw the keypoint label next to the keypoint
        ctx.fillText(
          key, // The keypoint name (e.g., "nose", "earL")
          coords[0] + 5, // Position the text slightly to the right of the keypoint
          coords[1] - 5 // Position the text slightly above the keypoint
        );
      }
      else { // if the keypoint visible 
        ctx.fillStyle = "#4eff10"; // Set the fill color for the keypoint to yellow #4eff10
        ctx.beginPath(); // Begin a new path for the keypoint circle
        ctx.arc(
          coords[0], // X-coordinate of the keypoint
          coords[1], // Y-coordinate of the keypoint
          5, // Radius of the keypoint circle
          0, // Start angle (0 radians)
          2 * Math.PI // End angle (full circle)
        );
        ctx.fill(); // Fill the keypoint circle with the current fill color
  
        // Draw the keypoint label next to the keypoint
        ctx.fillText(
          key, // The keypoint name (e.g., "nose", "earL")
          coords[0] + 5, // Position the text slightly to the right of the keypoint
          coords[1] - 5 // Position the text slightly above the keypoint
        );
      }

    }
  });
  // Refresh the annotation list to reflect the restored state
  updateAnnotationList();
}



// Function to save annotations for all images to the annotation.json file
async function saveAnnotations() {
  // Get the name of the currently displayed image
  const imageName = images[currentImageIndex].name;

  // --- Step 1: Sort Keypoints for Consistency ---
  if (annotations[imageName]) {
    // If there are annotations for the current image
    annotations[imageName].forEach(annotation => {
      // Create a new object to store keypoints in the desired order
      const sortedKeypoints = {};

      // Iterate over the predefined `keypoints` order (e.g., "nose", "earL", "earR", "tailB")
      keypoints.forEach(key => {
        // If the keypoint exists in the annotation, add it to the sortedKeypoints object
        if (key in annotation.keypoints) {
          sortedKeypoints[key] = annotation.keypoints[key];
        }
      });

      // Replace the original `keypoints` object with the sorted one
      annotation.keypoints = sortedKeypoints;
    });
  }

  // --- Step 2: Save Annotations to annotation.json ---
  const writable = await annotationFileHandle.createWritable(); // Open a writable stream to the annotation file
  await writable.write(JSON.stringify(annotations, null, 2)); // Write the annotations object as a pretty-printed JSON string
  await writable.close(); // Close the writable stream to save changes
}



// Function to update the annotation list displayed on the right side of the UI
function updateAnnotationList() {
  // Get the name of the currently displayed image
  const imageName = images[currentImageIndex].name;

  // Retrieve the annotations for the current image, defaulting to an empty array if none exist
  const imageAnnotations = annotations[imageName] || [];

  // Clear the existing list of annotations in the UI to prepare for the updated content
  annotationList.innerHTML = "";

  // Iterate over all annotations for the current image
  imageAnnotations.forEach((annotation, index) => {
    // Create a new `div` element to represent an annotation in the list
    const annotationItem = document.createElement("div");
    annotationItem.className = "annotationItem"; // Assign a CSS class for styling

    // Format the keypoints for display as "key: (x, y)"
    const keypointsDisplay = Object.entries(annotation.keypoints) // Convert keypoints object to an array of [key, coords]
      .map(([key, coords]) => `${key}: (${coords[0]}, ${coords[1]}, ${coords[2]}) <button onclick='rmKeypoint(${index}, "${key}")'>-</button>`) // Format each keypoint's name and coordinates
      .join("  <br />"); // Join all keypoints with line breaks for display

    // Set the inner HTML of the annotation item, including:
    // 1. Bounding box index
    // 2. Keypoints display
    // 3. A "Delete" button for removing the annotation
    annotationItem.innerHTML = `
      <span>Bounding Box ${index + 1}</span>
      <button onclick="selectBoundingBox(${index})">Select</button><br />
      <button onclick="deleteAnnotation(${index})">Delete BBox</button><br />
      ${keypointsDisplay}
  
    `;

    // // Add a mouseover event to highlight the bounding box on the canvas
    // annotationItem.addEventListener("mouseover", () => highlightBoundingBox(index));

    // // Add a mouseover event to highlight the bounding box on the canvas
    // annotationItem.addEventListener("click", () => selectBoundingBox(index));

    // Append the annotation item to the annotation list container
    annotationList.appendChild(annotationItem);
  });
}



// Function to remove a specific keypoint of a BBox
function rmKeypoint(index, key) {
  // Get the name of the currently displayed image
  const imageName = images[currentImageIndex].name;

  delete annotations[imageName][index]['keypoints'][key];


    // --- Save Updated Annotations ---
  // Save the modified annotations back to the annotation.json file
  saveAnnotations();

  // --- Update the Annotation List ---
  // Refresh the annotation list in the sidebar to reflect the deletion
  updateAnnotationList();

  // --- Redraw the Canvas ---
  // Reload the current image and redraw its annotations to reflect the changes
  loadImage();

  // --- Update File List and Progress ---
  // Refresh the file list on the left to update the annotation status for this image
  updateFileListAndProgress();

}


function selectBoundingBox(index) {

  selectedBBoxIndex = index;

  highlightBoundingBox(index);

  updateSelectedBBoxDisplay(index);

}


// Function to visually highlight a specific bounding box on the canvas
function highlightBoundingBox(index) {
  // Get the name of the currently displayed image
  const imageName = images[currentImageIndex].name;

  // Retrieve annotations for the current image, defaulting to an empty array if none exist
  const imageAnnotations = annotations[imageName] || [];

  // Get the specific annotation (bounding box) by its index
  const annotation = imageAnnotations[index];

  // If the annotation doesn't exist (e.g., invalid index), exit the function
  if (!annotation) return;

  // Extract the bounding box coordinates from the annotation
  const { bbox, keypoints } = annotation;

  // --- Draw Highlighted Bounding Box ---
  ctx.save(); // Save the current canvas state
  ctx.strokeStyle = "yellow"; // Set the stroke color to blue for highlighting
  ctx.lineWidth = 4; // Use a thicker line for better visibility
  ctx.strokeRect(
    bbox.x1, // X-coordinate of the top-left corner
    bbox.y1, // Y-coordinate of the top-left corner
    bbox.x2 - bbox.x1, // Width of the bounding box
    bbox.y2 - bbox.y1 // Height of the bounding box
  );

  for (const [key, coords] of Object.entries(keypoints)) {

      ctx.fillStyle =  "yellow"; // Set the fill color for the keypoint to yellow
      ctx.beginPath(); // Begin a new path for the keypoint circle
      ctx.arc(
        coords[0], // X-coordinate of the keypoint
        coords[1], // Y-coordinate of the keypoint
        5, // Radius of the keypoint circle
        0, // Start angle (0 radians)
        2 * Math.PI // End angle (full circle)
      );
      ctx.fill(); // Fill the keypoint circle with the current fill color

      // Draw the keypoint label next to the keypoint
      ctx.fillText(
        key, // The keypoint name (e.g., "nose", "earL")
        coords[0] + 5, // Position the text slightly to the right of the keypoint
        coords[1] - 5 // Position the text slightly above the keypoint
      );
  }
  ctx.restore(); // Restore the canvas to its previous state

  // --- Temporary Highlight Duration ---
  setTimeout(() => {
    // Clear the entire canvas to remove the highlight
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Reload the current image and its annotations to redraw them without the highlight
    loadImage();
  }, 1000); // Highlight duration is 1 second (1000 milliseconds)
}



// Function to delete an annotation for the current image
function deleteAnnotation(index) {
  // Get the name of the currently displayed image
  const imageName = images[currentImageIndex].name;

  // --- Backup for Undo ---
  // Save the current state of annotations to the undo stack before making changes
  undoStack.push(JSON.stringify(annotations));

  // --- Delete the Annotation ---
  // Remove the annotation at the specified index from the list of annotations for this image
  annotations[imageName].splice(index, 1);

  // --- Save Updated Annotations ---
  // Save the modified annotations back to the annotation.json file
  saveAnnotations();

  // --- Update the Annotation List ---
  // Refresh the annotation list in the sidebar to reflect the deletion
  updateAnnotationList();

  // --- Redraw the Canvas ---
  // Reload the current image and redraw its annotations to reflect the changes
  loadImage();

  // --- Update File List and Progress ---
  // Refresh the file list on the left to update the annotation status for this image
  updateFileListAndProgress();
}



// Event listener for the "Previous" button to navigate to the previous image
prevBtn.addEventListener("click", () => {
  // Check if there is a previous image in the `images` array
  if (currentImageIndex > 0) {
    // Decrease the current image index to move to the previous image
    currentImageIndex--;

    // Load the newly selected image and its annotations onto the canvas
    loadImage();

    // Update the file list on the left side to highlight the newly selected image
    updateFileListAndProgress();
  }
});



// Event listener for the "Next" button to navigate to the next image
nextBtn.addEventListener("click", () => {
  // Check if there is a next image in the `images` array
  if (currentImageIndex < images.length - 1) {
    // Increase the current image index to move to the next image
    currentImageIndex++;

    // Load the newly selected image and its annotations onto the canvas
    loadImage();

    // Update the file list on the left side to highlight the newly selected image
    updateFileListAndProgress();
  }
});



// Event listener to handle keyboard shortcuts for mode switching, saving, and undoing changes
document.addEventListener("keydown", (event) => {
  // --- Mode Switching (Ctrl + Alt) ---
  if (event.ctrlKey && event.altKey) {
    // If the current mode is "bbox" (bounding box), switch to "keypoint" mode
    if (mode === "bbox") {
      mode = "keypoint"; // Enter keypoint mode
    } else {
      // Cycle through keypoints in "keypoint" mode
      currentKeypointIndex = (currentKeypointIndex + 1) % keypoints.length;

      // If we've cycled through all keypoints, switch back to "bbox" mode
      if (currentKeypointIndex === 0) {
        mode = "bbox";
      }
    }

    // Update the UI to display the current mode
    updateModeDisplay();

    // Prevent default browser behavior (e.g., conflicts with built-in shortcuts)
    event.preventDefault();
  }
  // ---- Change 
  else if (event.ctrlKey && event.key === "x") { 
    
    if (visibleKeypoint) {
      visibleKeypoint = false;
    } else {
      visibleKeypoint = true;
    }

    // Update the UI to display the current mode
    updateModeDisplay()

    // Prevent default browser behavior (e.g., conflicts with built-in shortcuts)
    event.preventDefault();
  }
  // --- Save Annotations (Ctrl + S) ---
  else if (event.ctrlKey && event.key === "s") {
    // Prevent the browser's default save dialog
    event.preventDefault();

    // Save the current annotations to the annotation.json file
    saveAnnotations();

    // Notify the user that annotations have been saved
    alert("Annotations saved!");
  }
  // --- Undo Last Change (Ctrl + Z) ---
  else if (event.ctrlKey && event.key === "z") {
    // Check if there is a previous state in the undo stack
    if (undoStack.length > 0) {
      // Restore the last saved state from the undo stack
      annotations = JSON.parse(undoStack.pop());

      // Refresh the annotation list to reflect the restored state
      updateAnnotationList();

      // Reload the current image and its annotations on the canvas
      loadImage();
    }
  }
});



// Event listener for mouse down events on the canvas
canvas.addEventListener("mousedown", (event) => {
  // --- Get Mouse Coordinates Relative to the Canvas ---
  const rect = canvas.getBoundingClientRect(); // Get the canvas's position and dimensions
  const x = event.clientX - rect.left; // X-coordinate of the mouse relative to the canvas
  const y = event.clientY - rect.top; // Y-coordinate of the mouse relative to the canvas

  // --- Handle Interaction Based on Current Mode ---
  if (mode === "bbox") {
    // --- Bounding Box Mode ---
    // Start defining a bounding box by storing the starting (top-left) coordinates
    currentBBox = { x1: x, y1: y, x2: x, y2: y };
    dragging = true; // Set the dragging state to true to indicate the user is drawing a box
  } else if (mode === "keypoint") {
    // --- Keypoint Mode ---
    const selectedKeypoint = keypoints[currentKeypointIndex]; // Get the current keypoint being placed
    const imageName = images[currentImageIndex].name; // Get the name of the current image

    // Check if there are existing annotations for the current image
    if (!annotations[imageName]) return;

    // --- Find the Relevant Bounding Box ---
    // Select the last bounding box in the annotations for simplicity (can be enhanced for explicit selection)
    const selectedBBox = annotations[imageName][selectedBBoxIndex];

    // If a bounding box exists, add the keypoint to it
    if (selectedBBox) {
      if (isPointInBox(x, y, selectedBBox.bbox.x1, selectedBBox.bbox.y1, selectedBBox.bbox.x2, selectedBBox.bbox.y2)) {
        undoStack.push(JSON.stringify(annotations)); // Save the current state to the undo stack
        selectedBBox.keypoints[selectedKeypoint] = [x, y, (visibleKeypoint) ? 2 : 1]; // Add the keypoint with its coordinates

        drawAnnotations(); // Redraw the canvas to display the updated annotations
        saveAnnotations(); // Save the updated annotations to the annotation file
      }
    }
  }
});

function isPointInBox(x, y, x1, y1, x2, y2) {
  return (
    x >= x1 && x <= x2 &&
    y >= y1 && y <= y2
  );
}


// Event listener for mouse movement over the canvas
canvas.addEventListener("mousemove", (event) => {
  // --- Handle Bounding Box Drawing ---
  // Only proceed if the mode is "bbox" and the user is dragging (i.e., creating a bounding box)
  if (mode === "bbox" && dragging) {
    // --- Get Mouse Coordinates Relative to the Canvas ---
    const rect = canvas.getBoundingClientRect(); // Get the canvas's position and dimensions
    const x = event.clientX - rect.left; // X-coordinate of the mouse relative to the canvas
    const y = event.clientY - rect.top; // Y-coordinate of the mouse relative to the canvas

    // --- Update the Current Bounding Box ---
    // Update the bottom-right corner (`x2`, `y2`) of the bounding box as the mouse moves
    currentBBox.x2 = x;
    currentBBox.y2 = y;

    // --- Redraw the Canvas ---
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the entire canvas to remove previous drawings

    // Reload the current image to restore it after clearing the canvas
    loadImage();

    // Redraw existing annotations (bounding boxes and keypoints) for the current image
    drawAnnotations();

    // Draw the bounding box being created (the one under the mouse)
    drawCurrentBBox();
  }
});



// Event listener for mouse up events on the canvas
canvas.addEventListener("mouseup", () => {
  // --- Finalize Bounding Box Drawing ---
  // Only proceed if the mode is "bbox" and the user is dragging (i.e., creating a bounding box)
  if (mode === "bbox" && dragging) {
    dragging = false; // End the dragging state, indicating the bounding box is complete

    check_xFlag = currentBBox.x1 != currentBBox.x2
    check_yFlag = currentBBox.y1 != currentBBox.y2

    // check 
    if (check_xFlag && check_yFlag) {
      // --- Save Bounding Box to Annotations ---
      const imageName = images[currentImageIndex].name; // Get the name of the currently displayed image

      // Ensure there is an entry for the current image in the annotations object
      if (!annotations[imageName]) {
        annotations[imageName] = []; // Initialize an empty array for the image's annotations
      }

      finalBBox = { x1: 0, y1: 0, x2: 0, y2: 0};
      // making sure x1, y1 top left and x2, y2 bottom right
      finalBBox.x1 = currentBBox.x1 < currentBBox.x2 ? currentBBox.x1 : currentBBox.x2;
      finalBBox.y1 = currentBBox.y1 < currentBBox.y2 ? currentBBox.y1 : currentBBox.y2;
      finalBBox.x2 = currentBBox.x1 > currentBBox.x2 ? currentBBox.x1 : currentBBox.x2;
      finalBBox.y2 = currentBBox.y1 > currentBBox.y2 ? currentBBox.y1 : currentBBox.y2;

      // Add the newly created bounding box to the annotations for the current image
      annotations[imageName].push({
        bbox: finalBBox, // Add the bounding box coordinates
        keypoints: {}, // Initialize an empty keypoints object (to be populated later)
      });


      // --- Save Updated Annotations ---
      saveAnnotations(); // Persist the updated annotations to the annotation.json file
      // --- Reset State ---
      currentBBox = null; // Clear the temporary bounding box
      // --- Redraw the Canvas ---
      drawAnnotations(); // Refresh the canvas to include the newly added bounding box
      //Updated the selected Bounding Box
      selectBoundingBox(annotations[imageName].length - 1)
    } else {
      // --- Reset State ---
      currentBBox = null; // Clear the temporary bounding box

      // --- Redraw the Canvas ---
      ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the entire canvas to remove previous drawings

      // Reload the current image to restore it after clearing the canvas
      loadImage();

      // Redraw existing annotations (bounding boxes and keypoints) for the current image
      drawAnnotations();

    }
  }
});


// Function to draw the current bounding box being created while the user is dragging
function drawCurrentBBox() {
  // --- Check if a Bounding Box is Being Created ---
  // If there is no active bounding box (e.g., user is not dragging), exit the function
  if (!currentBBox) return;

  // --- Set Drawing Styles ---
  ctx.strokeStyle = "green"; // Set the color of the bounding box to green for real-time feedback
  ctx.lineWidth = 2; // Use a thin line for better visibility while dragging

  // --- Draw the Bounding Box ---
  ctx.strokeRect(
    currentBBox.x1, // X-coordinate of the top-left corner
    currentBBox.y1, // Y-coordinate of the top-left corner
    currentBBox.x2 - currentBBox.x1, // Width of the bounding box (difference between x2 and x1)
    currentBBox.y2 - currentBBox.y1 // Height of the bounding box (difference between y2 and y1)
  );
}