const folderInput = document.getElementById("folderInput");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const annotationList = document.getElementById("annotationItems");
const canvas = document.getElementById("annotationCanvas");
const ctx = canvas.getContext("2d");

let images = [];
let currentImageIndex = 0;
let annotations = {};
let currentBBox = null;
let dragging = false;
let mode = "bbox"; // Default mode
let currentKeypointIndex = 0;
const keypoints = ["nose", "earL", "earR", "tailB"];
let annotationFileHandle = null;
let undoStack = [];

// Update the display mode
function updateModeDisplay() {
  document.getElementById("modeDisplay").textContent =
    mode === "bbox" ? "Mode: Bounding Box" : `Mode: Keypoint (${keypoints[currentKeypointIndex]})`;
}

function updateFileListAndProgress() {
    const fileList = document.getElementById("fileList");
    fileList.innerHTML = ""; // Clear the file list
  
    images.forEach((image, index) => {
      const imageName = image.name;
      const imageAnnotations = annotations[imageName] || [];
  
      // Check if the frame is fully annotated
      const isComplete =
        imageAnnotations.length === 5 &&
        imageAnnotations.every(annotation => Object.keys(annotation.keypoints).length === 4);
  
      // Check if the frame is partially annotated
      const isPartial = imageAnnotations.length > 0 && !isComplete;
  
      // Create file list entry
      const fileItem = document.createElement("div");
      fileItem.textContent = imageName;
      fileItem.className = index === currentImageIndex ? "selected" : "";
  
      // Assign color based on annotation completeness
      if (isComplete) {
        fileItem.style.color = "green"; // Green for fully annotated frames
      } else if (isPartial) {
        fileItem.style.color = "orange"; // Orange for partially annotated frames
      } else {
        fileItem.style.color = "gray"; // Gray for frames with no annotations
      }
  
      // Add a click event to navigate to the selected image
      fileItem.addEventListener("click", () => {
        currentImageIndex = index;
        loadImage();
        updateFileListAndProgress(); // Update the list to highlight the selected file
      });
  
      fileList.appendChild(fileItem);
    });
  }
  

// Load a folder
folderInput.addEventListener("click", async () => {
  const directoryHandle = await window.showDirectoryPicker();
  images = [];
  annotations = {};

  for await (const entry of directoryHandle.values()) {
    if (entry.kind === "file" && /\.(png|jpe?g)$/i.test(entry.name)) {
      images.push({ name: entry.name, handle: entry });
    }
    if (entry.name === "annotation.json") {
      annotationFileHandle = entry;
    }
  }

  images.sort((a, b) => a.name.localeCompare(b.name));

  if (!annotationFileHandle) {
    annotationFileHandle = await directoryHandle.getFileHandle("annotation.json", {
      create: true,
    });
    saveAnnotations(); // Create an empty JSON file
  } else {
    const file = await annotationFileHandle.getFile();
    annotations = JSON.parse(await file.text());
  }

  currentImageIndex = 0;
  loadImage();
  updateFileListAndProgress();
});

// Load the current image
async function loadImage() {
  if (images.length === 0) return;

  const imageFile = await images[currentImageIndex].handle.getFile();
  const imageURL = URL.createObjectURL(imageFile);

  const img = new Image();
  img.src = imageURL;
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    drawAnnotations();
  };

  updateAnnotationList();
}

// Draw annotations on the canvas
function drawAnnotations() {
  const imageName = images[currentImageIndex].name;
  const imageAnnotations = annotations[imageName] || [];

  imageAnnotations.forEach(({ bbox, keypoints }, index) => {
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      bbox.x1,
      bbox.y1,
      bbox.x2 - bbox.x1,
      bbox.y2 - bbox.y1
    );

    for (const [key, coords] of Object.entries(keypoints)) {
      ctx.fillStyle = "yellow";
      ctx.beginPath();
      ctx.arc(coords[0], coords[1], 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillText(key, coords[0] + 5, coords[1] - 5);
    }
  });
}

// Save annotations to the JSON file
async function saveAnnotations() {
  const writable = await annotationFileHandle.createWritable();
  await writable.write(JSON.stringify(annotations, null, 2));
  await writable.close();

  updateFileListAndProgress();
}

// Update the annotation list on the right side
function updateAnnotationList() {
  const imageName = images[currentImageIndex].name;
  const imageAnnotations = annotations[imageName] || [];

  annotationList.innerHTML = "";

  imageAnnotations.forEach((annotation, index) => {
    const annotationItem = document.createElement("div");
    annotationItem.className = "annotationItem";

    const keypointsDisplay = Object.entries(annotation.keypoints)
      .map(([key, coords]) => `${key}: (${coords[0]}, ${coords[1]})`)
      .join("<br>");

    annotationItem.innerHTML = `
      <span>Bounding Box ${index + 1}</span><br>
      ${keypointsDisplay}
      <button onclick="deleteAnnotation(${index})">Delete</button>
    `;

    annotationItem.addEventListener("mouseover", () => highlightBoundingBox(index));

    annotationList.appendChild(annotationItem);
  });
}

// Highlight a bounding box
function highlightBoundingBox(index) {
  const imageName = images[currentImageIndex].name;
  const imageAnnotations = annotations[imageName] || [];
  const annotation = imageAnnotations[index];

  if (!annotation) return;

  const { bbox } = annotation;
  ctx.save();
  ctx.strokeStyle = "blue";
  ctx.lineWidth = 4;
  ctx.strokeRect(
    bbox.x1,
    bbox.y1,
    bbox.x2 - bbox.x1,
    bbox.y2 - bbox.y1
  );
  ctx.restore();

  setTimeout(() => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    loadImage();
  }, 1000);
}

// Delete an annotation
function deleteAnnotation(index) {
  const imageName = images[currentImageIndex].name;
  undoStack.push(JSON.stringify(annotations));
  annotations[imageName].splice(index, 1);
  saveAnnotations();
  updateAnnotationList();
  loadImage();

  updateFileListAndProgress();
}

// Add navigation logic
prevBtn.addEventListener("click", () => {
  if (currentImageIndex > 0) {
    currentImageIndex--;
    loadImage();
    updateFileListAndProgress();

  }
});

nextBtn.addEventListener("click", () => {
  if (currentImageIndex < images.length - 1) {
    currentImageIndex++;
    loadImage();
    updateFileListAndProgress();
  }
});

// Handle key events for mode switching and saving
document.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.altKey) {
    if (mode === "bbox") {
      mode = "keypoint";
    } else {
      currentKeypointIndex = (currentKeypointIndex + 1) % keypoints.length;
      if (currentKeypointIndex === 0) {
        mode = "bbox";
      }
    }
    updateModeDisplay();
    event.preventDefault(); // Prevent browser conflicts
  } else if (event.ctrlKey && event.key === "s") {
    event.preventDefault();
    saveAnnotations();
    alert("Annotations saved!");
  } else if (event.ctrlKey && event.key === "z") {
    if (undoStack.length > 0) {
      annotations = JSON.parse(undoStack.pop());
      updateAnnotationList();
      loadImage();
    }
  }
});

// Canvas interaction
canvas.addEventListener("mousedown", (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  if (mode === "bbox") {
    // Start drawing a bounding box
    currentBBox = { x1: x, y1: y, x2: x, y2: y };
    dragging = true;
  } else if (mode === "keypoint") {
    // Add a keypoint
    const selectedKeypoint = keypoints[currentKeypointIndex];
    const imageName = images[currentImageIndex].name;
    if (!annotations[imageName]) return;

    // Find the last bounding box (or implement explicit selection logic)
    const lastBBox = annotations[imageName][annotations[imageName].length - 1];
    if (lastBBox) {
      undoStack.push(JSON.stringify(annotations));
      lastBBox.keypoints[selectedKeypoint] = [x, y];
      drawAnnotations();
      saveAnnotations();
    }
  }
});

canvas.addEventListener("mousemove", (event) => {
  if (mode === "bbox" && dragging) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Update the bounding box as the mouse moves
    currentBBox.x2 = x;
    currentBBox.y2 = y;

    // Redraw canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    loadImage();
    drawAnnotations();
    drawCurrentBBox();
  }
});

canvas.addEventListener("mouseup", () => {
  if (mode === "bbox" && dragging) {
    dragging = false;

    const imageName = images[currentImageIndex].name;
    if (!annotations[imageName]) {
      annotations[imageName] = [];
    }
    annotations[imageName].push({ bbox: currentBBox, keypoints: {} });
    saveAnnotations();
    currentBBox = null;

    drawAnnotations();
  }
});

// Draw the current bounding box while dragging
function drawCurrentBBox() {
  if (!currentBBox) return;

  ctx.strokeStyle = "green";
  ctx.lineWidth = 2;
  ctx.strokeRect(
    currentBBox.x1,
    currentBBox.y1,
    currentBBox.x2 - currentBBox.x1,
    currentBBox.y2 - currentBBox.y1
  );
}