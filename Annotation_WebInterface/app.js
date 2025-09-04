/* ====================================================================== *
 *  BehavTrack — Minimal Web Annotator (Bounding Boxes + Keypoints)
 *  - Single-image canvas UI with keyboard + mouse interactions
 *  - Stores annotations in annotation.json inside the selected folder
 *  - Status panel shows per-frame completion + quick sort toggle
 * ====================================================================== */

// ---- DOM references ----------------------------------------------------
const folderInput = document.getElementById("folderInput");           // Folder picker button
const prevBtn = document.getElementById("prevBtn");                   // Go to previous image
const nextBtn = document.getElementById("nextBtn");                   // Go to next image
const annotationList = document.getElementById("annotationItems");    // Right sidebar annotation list
const canvas = document.getElementById("annotationCanvas");           // Main drawing canvas
const ctx = canvas.getContext("2d");                                  // 2D drawing context

// ---- Global state ------------------------------------------------------
// NOTE: This app is deliberately simple—state is kept in-memory and
// synchronized to disk (annotation.json) after edits to avoid losing work.
let images = [];                    // [{ name, handle }, ...] from selected folder
let currentImageIndex = 0;          // Index into `images`
let annotations = {};               // { [imageName]: [{ bbox, keypoints, mAnnotated }, ...], ... }
let currentBBox = null;             // Temporary bbox while dragging: { x1,y1,x2,y2 } or null
let dragging = false;               // True while mouse drag to create bbox
let mode = "bbox";                  // "bbox" | "keypoint"
let currentKeypointIndex = 0;       // Index into `keypoints` when placing keypoints
const keypoints = ["nose", "earL", "earR", "tailB"]; // Ordered schema; used for sorting/export
let annotationFileHandle = null;    // FileHandle for annotation.json in the chosen folder
let undoStack = [];                 // Simple undo buffer (stores JSON snapshots); disabled by default
let visibleKeypoint = false;        // Toggle for setting kp visibility flag: false → 1, true → 2
let selectedBBoxIndex = 0;          // Which bbox is "selected" for keypoint placement
let framesComplete = [];            // Names of images with fully completed manual annotations
let sort_notComplete = false;       // Toggle: sort incomplete frames to the top

/* ====================================================================== *
 *  UI helpers
 * ====================================================================== */

/**
 * Update the toolbar mode indicator.
 * Shows either: "Bounding Box" or "Keypoint (<name>) - Visibility (<bool>)"
 */
function updateModeDisplay() {
  document.getElementById("modeDisplay").innerHTML =
    mode === "bbox"
      ? "Mode: <span id='mode_bbox'> Bounding Box </span>"
      : `Mode: Keypoint (<span id='mode_${keypoints[currentKeypointIndex]}'>${keypoints[currentKeypointIndex]}</span>) - Visibility (<span id='visibility_${visibleKeypoint}'>${visibleKeypoint}</span>)`;
}

/**
 * Update the "selected bbox" info block (count + selected index).
 */
function updateSelectedBBoxDisplay() {
  const imageName = images[currentImageIndex].name;
  const imageAnnotations = annotations[imageName] || [];
  document.getElementById("bBox").innerHTML =
    `Annotations: ${imageAnnotations.length} <br /> Selected Bounding Box: ${selectedBBoxIndex + 1}`;
}

/**
 * A frame is "complete" if:
 *  - exactly 5 bounding boxes
 *  - each has exactly 4 keypoints
 *  - all are marked as manually annotated (`mAnnotated: true`)
 */
function imageAnnotations_isComplete(imageName) {
  const imageAnnotations = annotations[imageName] || [];
  return (
    imageAnnotations.length === 5 &&
    imageAnnotations.every(a => Object.keys(a.keypoints).length === 4) &&
    imageAnnotations.every(a => a.mAnnotated)
  );
}

/**
 * Sort file list either alphabetically or by "not complete first".
 * Keeps currentImageIndex pointing to the same image object after sort.
 */
function performSort_fileListAndProgress() {
  sort_notComplete = !sort_notComplete;

  // Sort comparator: completed frames go last; incomplete first.
  const imgSortFn = (a, b) => {
    if (imageAnnotations_isComplete(a.name) && !imageAnnotations_isComplete(b.name)) return 1;
    if (!imageAnnotations_isComplete(a.name) && imageAnnotations_isComplete(b.name)) return -1;
    return 0;
  };

  const currentImage_obj = images[currentImageIndex]; // preserve reference

  if (sort_notComplete) {
    images.sort(imgSortFn);
  } else {
    images.sort((a, b) => a.name.localeCompare(b.name));
  }

  // Restore index of current image after sort
  currentImageIndex = images.indexOf(currentImage_obj);

  loadImage();                 // Re-render canvas with the same image
  updateFileListAndProgress(); // Refresh sidebar
}

/**
 * Rebuild the file list sidebar and progress header.
 * Colors:
 *  - green:   fully annotated (and manually annotated)
 *  - blue:    contains predictions (mAnnotated=false somewhere)
 *  - orange:  partially annotated (some boxes/kpts missing)
 *  - black:   no annotations yet
 */
function updateFileListAndProgress() {
  const fileList = document.getElementById("fileList");
  fileList.innerHTML = "";

  images.forEach((image, index) => {
    const imageName = image.name;
    const imageAnnotations = annotations[imageName] || [];

    const isComplete =
      imageAnnotations.length === 5 &&
      imageAnnotations.every(a => Object.keys(a.keypoints).length === 4);

    // "noPredictions" means all are marked manual (true)
    const noPredictions = imageAnnotations.every(a => a.mAnnotated);

    // Create clickable line item
    const fileItem = document.createElement("div");
    fileItem.textContent = imageName;
    fileItem.className = index === currentImageIndex ? "selected" : "";

    // Maintain framesComplete list + color by state
    if (isComplete && noPredictions) {
      if (!framesComplete.includes(imageName)) framesComplete.push(imageName);
      fileItem.style.color = "green";
    } else {
      const pos = framesComplete.indexOf(imageName);
      if (pos !== -1) framesComplete.splice(pos, 1);

      if (!noPredictions) {
        fileItem.style.color = "blue";   // predicted content present
      } else if (imageAnnotations.length > 0 && !isComplete) {
        fileItem.style.color = "orange"; // partial manual
      } else {
        fileItem.style.color = "black";  // none
      }
    }

    // Progress header (top of list)
    document.getElementById("frameProgress").innerHTML = `
      <span>Frames: ${framesComplete.length}/${images.length}</span>
      <button onclick='performSort_fileListAndProgress()'>sort</button>
    `;

    // Clicking a file moves to that image
    fileItem.addEventListener("click", () => {
      currentImageIndex = index;
      loadImage();
      updateFileListAndProgress();
    });

    fileList.appendChild(fileItem);
  });
}

/* ====================================================================== *
 *  Folder loading
 * ====================================================================== */

/**
 * Folder picker — loads images + finds/creates annotation.json.
 * Uses File System Access API (works in Chromium-based browsers with HTTPS).
 */
folderInput.addEventListener("click", async () => {
  // Ask user to pick a directory
  const directoryHandle = await window.showDirectoryPicker();

  // Reset in-memory state
  images = [];
  annotations = {};
  annotationFileHandle = null;

  // Populate images[] and find annotation.json if present
  for await (const entry of directoryHandle.values()) {
    if (entry.kind === "file" && /\.(png|jpe?g)$/i.test(entry.name)) {
      images.push({ name: entry.name, handle: entry });
    }
    if (entry.name === "annotation.json") {
      annotationFileHandle = entry;
    }
  }

  // Sort by filename (stable order)
  images.sort((a, b) => a.name.localeCompare(b.name));

  // Ensure annotation.json exists
  if (!annotationFileHandle) {
    annotationFileHandle = await directoryHandle.getFileHandle("annotation.json", { create: true });
    // initialize empty map on first run
    await saveAnnotations(); // writes `{}` initially
  } else {
    // Load prior annotations from disk
    const file = await annotationFileHandle.getFile();
    annotations = JSON.parse(await file.text());
  }

  // Start at the first image
  currentImageIndex = 0;

  // Render first image + sidebars
  loadImage();
  updateFileListAndProgress();
});

/* ====================================================================== *
 *  Canvas rendering
 * ====================================================================== */

/**
 * Load the currently selected image into the canvas, then draw annotations.
 * Uses a blob URL to avoid reading the entire file into memory.
 */
async function loadImage() {
  if (images.length === 0) return;

  const imageFile = await images[currentImageIndex].handle.getFile();
  const imageURL = URL.createObjectURL(imageFile);

  const img = new Image();
  img.src = imageURL;

  img.onload = () => {
    // Fit canvas to image (pixel-accurate)
    canvas.width = img.width;
    canvas.height = img.height;

    // Reset and draw background image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    // Overlay stored annotations
    drawAnnotations();
  };

  // Refresh right sidebar list
  updateAnnotationList();
}

/**
 * Draw all annotations (bboxes + keypoints) for the current image.
 * - Red boxes:   mAnnotated == true (manual)
 * - Blue boxes:  mAnnotated == false (predictions)
 * - Visible kp:  green-ish; Invisible kp: light/amber
 */
function drawAnnotations() {
  const imageName = images[currentImageIndex].name;
  const imageAnnotations = annotations[imageName] || [];

  imageAnnotations.forEach(({ bbox, keypoints, mAnnotated }, index) => {
    // --- Box ---
    ctx.strokeStyle = mAnnotated ? "red" : "blue";
    ctx.lineWidth = 2;
    ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

    // Index label near top-left
    ctx.font = "15px Arial";
    ctx.fillStyle = "#45f7dc";
    ctx.fillText(`${index + 1}`, bbox.x1 + 5, bbox.y1 - 5);

    // --- Keypoints ---
    ctx.font = "normal 10px Arial";
    for (const [key, coords] of Object.entries(keypoints)) {
      // coords = [x, y, visFlag]; visFlag: 1=invisible(weak), 2=visible
      const [kx, ky, v] = coords;

      if (v == 1) {
        ctx.fillStyle = mAnnotated ? "orange" : "#F0FFFF";
      } else {
        ctx.fillStyle = mAnnotated ? "#4eff10" : "#088F8F";
      }

      ctx.beginPath();
      ctx.arc(kx, ky, 5, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillText(key, kx + 5, ky - 5);
    }
  });

  // keep sidebars aligned with canvas state
  updateAnnotationList();
}

/* ====================================================================== *
 *  Persistence
 * ====================================================================== */

/**
 * Save the global `annotations` object to annotation.json.
 * Ensures keypoints in each bbox are sorted by the global `keypoints` order.
 */
async function saveAnnotations() {
  const imageName = images[currentImageIndex]?.name;

  // Sort keypoints for predictable file output
  if (imageName && annotations[imageName]) {
    annotations[imageName].forEach(annotation => {
      const sortedKeypoints = {};
      keypoints.forEach(k => {
        if (k in annotation.keypoints) sortedKeypoints[k] = annotation.keypoints[k];
      });
      annotation.keypoints = sortedKeypoints;
    });
  }

  // Persist to disk
  const writable = await annotationFileHandle.createWritable();
  await writable.write(JSON.stringify(annotations, null, 2));
  await writable.close();
}

/* ====================================================================== *
 *  Right sidebar (annotation list)
 * ====================================================================== */

/**
 * Rebuild the sidebar list for the current image: bboxes + per-kp rows.
 * Includes buttons to select a bbox, delete a bbox, and remove individual kps.
 */
function updateAnnotationList() {
  const imageName = images[currentImageIndex].name;
  const imageAnnotations = annotations[imageName] || [];

  annotationList.innerHTML = "";

  imageAnnotations.forEach((annotation, index) => {
    const annotationItem = document.createElement("div");
    annotationItem.className = "annotationItem";

    // Pretty-print keypoints with small "remove" buttons
    const keypointsDisplay = Object.entries(annotation.keypoints)
      .map(
        ([key, coords]) =>
          `${key}: (${coords[0]}, ${coords[1]}, ${coords[2]}) <button onclick='rmKeypoint(${index}, "${key}")'>-</button>`
      )
      .join("  <br />");

    annotationItem.innerHTML = `
      <span>Bounding Box ${index + 1}</span>
      <button onclick="selectBoundingBox(${index})">Select</button><br />
      <button onclick="deleteAnnotation(${index})">Delete BBox</button><br />
      ${keypointsDisplay}
    `;

    annotationList.appendChild(annotationItem);
  });

  updateFileListAndProgress();
  updateSelectedBBoxDisplay();
}

/**
 * Remove a single keypoint from a bbox (by bbox index + key name).
 * Then persist + refresh UI.
 */
function rmKeypoint(index, key) {
  const imageName = images[currentImageIndex].name;
  delete annotations[imageName][index].keypoints[key];

  saveAnnotations();
  updateAnnotationList();
  loadImage();
  updateFileListAndProgress();
}

/**
 * Set the currently selected bbox index and briefly highlight it on the canvas.
 */
function selectBoundingBox(index) {
  selectedBBoxIndex = index;
  highlightBoundingBox();
  updateSelectedBBoxDisplay();
}

/**
 * Mark the selected bbox as manually annotated (mAnnotated=true),
 * persist and refresh render + lists.
 */
function update_mAnnotated() {
  const imageName = images[currentImageIndex].name;
  const imageAnnotations = annotations[imageName] || [];
  const annotation = imageAnnotations[selectedBBoxIndex];
  if (!annotation) return;

  annotation.mAnnotated = true;

  saveAnnotations();
  updateAnnotationList();
  loadImage();
  updateFileListAndProgress();
}

/**
 * Briefly draws a highlighted bbox + yellow keypoints for the selected index,
 * then re-renders the image after a 1s timeout.
 */
function highlightBoundingBox() {
  const imageName = images[currentImageIndex].name;
  const imageAnnotations = annotations[imageName] || [];
  const annotation = imageAnnotations[selectedBBoxIndex];
  if (!annotation) return;

  // Auto-mark as manual if not yet set; keeps green/blue logic consistent
  if (!annotation.mAnnotated) update_mAnnotated();

  const { bbox, keypoints } = annotation;

  ctx.save();
  ctx.strokeStyle = "yellow";
  ctx.lineWidth = 4;
  ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

  ctx.font = "15px Arial";
  ctx.strokeText(selectedBBoxIndex + 1, bbox.x1 + 5, bbox.y1 - 5);

  ctx.font = "normal 10px Arial";
  for (const [key, coords] of Object.entries(keypoints)) {
    ctx.fillStyle = "yellow";
    ctx.beginPath();
    ctx.arc(coords[0], coords[1], 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillText(key, coords[0] + 5, coords[1] - 5);
  }
  ctx.restore();

  // Remove highlight after 1s by redrawing the image + annotations
  setTimeout(() => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    loadImage();
  }, 1000);
}

/**
 * Delete an entire bbox annotation (by index) from the current image.
 * Pushes a JSON snapshot to the `undoStack` (if you later enable undo).
 */
function deleteAnnotation(index) {
  const imageName = images[currentImageIndex].name;

  // Store snapshot for potential undo (ctrl+z is currently commented out)
  undoStack.push(JSON.stringify(annotations));

  // Remove bbox
  annotations[imageName].splice(index, 1);

  saveAnnotations();
  updateAnnotationList();
  loadImage();
  updateFileListAndProgress();
}

/* ====================================================================== *
 *  Navigation
 * ====================================================================== */

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

/* ====================================================================== *
 *  Keyboard shortcuts
 * ====================================================================== */
//  - Ctrl+Alt : toggle modes & cycle keypoints (bbox → keypoint1 → ... → bbox)
//  - Ctrl+X   : toggle visibility flag for new keypoints (false=1, true=2)
//  - Ctrl+S   : save annotations (writes annotation.json)

document.addEventListener("keydown", (event) => {
  // Mode toggle / cycle (Ctrl + Alt)
  if (event.ctrlKey && event.altKey) {
    if (mode === "bbox") {
      mode = "keypoint";
    } else {
      currentKeypointIndex = (currentKeypointIndex + 1) % keypoints.length;
      if (currentKeypointIndex === 0) mode = "bbox";
    }
    updateModeDisplay();
    event.preventDefault();
  }
  // Toggle visibility flag (Ctrl + X): 1 ↔ 2
  else if (event.ctrlKey && event.key === "x") {
    visibleKeypoint = !visibleKeypoint;
    updateModeDisplay();
    event.preventDefault();
  }
  // Save annotations (Ctrl + S)
  else if (event.ctrlKey && event.key === "s") {
    event.preventDefault(); // avoid browser save dialog
    saveAnnotations().then(() => alert("Annotations saved!"));
  }
  // Undo (Ctrl + Z) — optional
  // else if (event.ctrlKey && event.key === "z") {
  //   if (undoStack.length > 0) {
  //     annotations = JSON.parse(undoStack.pop());
  //     updateAnnotationList();
  //     loadImage();
  //   }
  // }
});

/* ====================================================================== *
 *  Mouse interactions on canvas
 * ====================================================================== */

/**
 * mousedown:
 *  - bbox mode: begin a new bbox at (x1,y1)
 *  - keypoint mode: add kp to the currently selected bbox (if inside bbox)
 */
canvas.addEventListener("mousedown", (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  if (mode === "bbox") {
    currentBBox = { x1: x, y1: y, x2: x, y2: y };
    dragging = true;
  } else if (mode === "keypoint") {
    const selectedKeypoint = keypoints[currentKeypointIndex];
    const imageName = images[currentImageIndex].name;

    if (!annotations[imageName]) return;

    const selectedBBox = annotations[imageName][selectedBBoxIndex];
    if (!selectedBBox) return;

    // Only add the kp if the click lies inside the selected bbox
    if (isPointInBox(x, y, selectedBBox.bbox.x1, selectedBBox.bbox.y1, selectedBBox.bbox.x2, selectedBBox.bbox.y2)) {
      undoStack.push(JSON.stringify(annotations));
      selectedBBox.keypoints[selectedKeypoint] = [x, y, visibleKeypoint ? 2 : 1];

      drawAnnotations();
      saveAnnotations();
    }
  }
});

/**
 * Basic AABB contains test for placing keypoints inside a bbox.
 */
function isPointInBox(x, y, x1, y1, x2, y2) {
  return x >= x1 && x <= x2 && y >= y1 && y <= y2;
}

/**
 * mousemove:
 *  - live-draw a bbox while dragging (in bbox mode).
 *    We redraw the image + all annotations to keep cursor feedback accurate.
 */
canvas.addEventListener("mousemove", (event) => {
  if (mode === "bbox" && dragging) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    currentBBox.x2 = x;
    currentBBox.y2 = y;

    // Repaint background and existing annotations
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    loadImage();    // async img load; redrawAnnotations is called in onload

    // Draw the in-progress bbox on top (immediate feedback)
    drawCurrentBBox();
  }
});

/**
 * mouseup:
 *  - finalize bbox if valid (x1!=x2 and y1!=y2); normalize coords (top-left/bottom-right)
 *  - persist + refresh selection + redraw
 */
canvas.addEventListener("mouseup", () => {
  if (mode === "bbox" && dragging) {
    dragging = false;

    const check_xFlag = currentBBox.x1 !== currentBBox.x2;
    const check_yFlag = currentBBox.y1 !== currentBBox.y2;

    if (check_xFlag && check_yFlag) {
      const imageName = images[currentImageIndex].name;
      if (!annotations[imageName]) annotations[imageName] = [];

      // Normalize bbox so that (x1,y1) is top-left and (x2,y2) bottom-right
      const finalBBox = {
        x1: Math.min(currentBBox.x1, currentBBox.x2),
        y1: Math.min(currentBBox.y1, currentBBox.y2),
        x2: Math.max(currentBBox.x1, currentBBox.x2),
        y2: Math.max(currentBBox.y1, currentBBox.y2),
      };

      annotations[imageName].push({
        bbox: finalBBox,
        keypoints: {},
        mAnnotated: true, // manual box
      });

      saveAnnotations();
      currentBBox = null;
      drawAnnotations();

      // auto-select the newly added box
      selectBoundingBox(annotations[imageName].length - 1);
    } else {
      // A zero-size drag → discard and fully redraw the base image + annots
      currentBBox = null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      loadImage();
      drawAnnotations();
    }
  }
});

/**
 * Draw a live "in-progress" bbox while dragging.
 */
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


/* ====================================================================== *
 *  Help panel toggle logic
 * ====================================================================== */

const helpToggle = document.getElementById("helpToggle");
const helpPanel  = document.getElementById("helpPanel");
const helpClose  = document.getElementById("helpClose");

function openHelp()   { helpPanel.classList.remove("hidden"); }
function closeHelp()  { helpPanel.classList.add("hidden"); }
function toggleHelp() { helpPanel.classList.toggle("hidden"); }

// Open/close via buttons
helpToggle.addEventListener("click", toggleHelp);
helpClose.addEventListener("click", closeHelp);

// Close when clicking the dimmed background (but not the content)
helpPanel.addEventListener("click", (e) => {
  if (e.target === helpPanel) closeHelp();
});

// Keyboard: 'H' toggles help (works alongside your other shortcuts)
document.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "h" && !event.ctrlKey && !event.altKey && !event.metaKey) {
    toggleHelp();
  }
});

