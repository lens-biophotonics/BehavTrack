# ui.py
"""
This module provides the main graphical user interface (GUI) for the Stratified Frame Selector application.
It uses PySide6 (Qt for Python) to present a window where users can:

1. Select a folder containing video files.
2. Configure parameters for stratified frame sampling:
    - Total frames to extract (n)
    - Number of clusters (k)
    - Frame skip interval
3. Estimate how many frames will be extracted based on the chosen parameters before running.
4. Run the stratified sampling in the background, showing progress updates directly in the UI.
5. After completion, browse through the selected frames, grouped by cluster.
6. Preview the selected frames (as images) on the right side of the UI.

The application integrates with the `frame_sampling` module for logic and performs all heavy work in a background thread
to keep the UI responsive.
"""

import sys
import os
import cv2
import numpy as np

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLineEdit, QLabel,
                               QFileDialog, QSpinBox, QTableWidget, QTableWidgetItem,
                               QHeaderView, QGroupBox, QFormLayout, QMessageBox, QSplitter, QComboBox, QProgressBar)
from PySide6.QtCore import Qt, QRunnable, QThreadPool, Signal, Slot, QObject
from PySide6.QtGui import QPixmap, QImage

from frame_sampling import stratified_frame_selection


class WorkerSignals(QObject):
    """
    WorkerSignals defines Qt signals that the background worker can emit to communicate with the UI thread.

    Signals:
    - finished: Emitted with a dict of results when the background task completes successfully.
    - error: Emitted with a string error message if an exception occurs in the background task.
    - progress: Emitted with an integer (0-100) to update the progress bar in the main UI.
    """
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)


class Worker(QRunnable):
    """
    Worker is a QRunnable that executes the stratified_frame_selection logic in a separate thread.

    By using a worker:
    - The main UI remains responsive during long-running tasks (e.g., processing many video files).
    - Progress can be reported back to the UI via signals.

    The worker is initialized with:
    - folder (str): the path to the folder containing video files.
    - n (int): total number of frames to select.
    - k (int): number of clusters.
    - frame_skip (int): interval at which frames are sampled from the videos.
    """

    def __init__(self, folder, n, k, frame_skip):
        super().__init__()
        self.folder = folder
        self.n = n
        self.k = k
        self.frame_skip = frame_skip
        self.signals = WorkerSignals()

    def run(self):
        """
        run() is called when the worker is started in the thread pool.
        It calls stratified_frame_selection and handles progress updates via the progress_callback.
        Any errors are caught and sent via the error signal.
        """
        try:
            # Define a callback to update progress, forwarded to main UI via signals.
            def progress_callback(val):
                self.signals.progress.emit(val)

            # Run the main logic in frame_sampling.py
            results = stratified_frame_selection(
                self.folder,
                n=self.n,
                k=self.k,
                frame_skip=self.frame_skip,
                progress_callback=progress_callback
            )
            # If successful, emit finished signal with the resulting dictionary.
            self.signals.finished.emit(results)
        except Exception as e:
            # If any error occurs, emit the error signal so the UI can show an error message.
            self.signals.error.emit(str(e))


class MainWindow(QMainWindow):
    """
    MainWindow is the primary UI class. It sets up:
    - A left panel with controls (folder selection, parameters, run button, progress bar, clusters combo, table of frames).
    - A right panel that displays a selected frame as an image.

    The UI flow:
    1. User selects a folder and sets parameters (n, k, frame_skip).
    2. The application estimates how many frames can be extracted and displays this info.
    3. When the user clicks "Run":
       - The parameters and folder are validated.
       - A Worker is created and started in a thread pool to run the selection process.
       - The UI shows a progress bar while the work is being done.
    4. After completion, the UI displays the selected frames in a table, sorted by video and clustered.
    5. The user can select a cluster from the combo box to filter displayed frames.
    6. Clicking a frame in the table shows that frame on the right panel.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stratified Frame Selector")

        # Thread pool for running background tasks without freezing the UI.
        self.thread_pool = QThreadPool()

        # Create a QSplitter to divide the UI into a left and right panel.
        splitter = QSplitter()
        left_widget = QWidget()
        right_widget = QWidget()

        # Add the two widgets to the splitter: left and right panels.
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        # Set stretch factors so the right panel is larger.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # The splitter is set as the main window's central widget.
        self.setCentralWidget(splitter)

        # LEFT PANEL SETUP
        left_layout = QVBoxLayout(left_widget)

        # PARAMETERS GROUP
        # This group box holds fields to select video folder, n, k, and frame_skip.
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()

        # Video Folder Selection:
        # A QLineEdit to show the selected folder path and a button to open a QFileDialog.
        self.folder_line_edit = QLineEdit()
        folder_button = QPushButton("Select Video Folder")
        folder_button.clicked.connect(self.select_folder)
        folder_h_layout = QHBoxLayout()
        folder_h_layout.addWidget(self.folder_line_edit)
        folder_h_layout.addWidget(folder_button)
        param_layout.addRow(QLabel("Video Folder:"), folder_h_layout)

        # N: Total frames to select
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 100000)
        self.n_spin.setValue(50)
        # When n changes, update the frame estimation to keep the user informed.
        self.n_spin.valueChanged.connect(self.update_frame_estimation)
        param_layout.addRow(QLabel("Total frames (n):"), self.n_spin)

        # K: Number of clusters
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 1000)
        self.k_spin.setValue(5)
        param_layout.addRow(QLabel("Number of clusters (k):"), self.k_spin)

        # Frame Skip: Interval to pick frames from the videos
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 1000)
        self.frame_skip_spin.setValue(30)
        self.frame_skip_spin.valueChanged.connect(self.update_frame_estimation)
        param_layout.addRow(QLabel("Frame skip:"), self.frame_skip_spin)

        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # Estimated Frames Label:
        # Shows the user how many frames will be extracted given current folder and frame_skip.
        self.estimated_frames_label = QLabel("Estimated frames: N/A")
        left_layout.addWidget(self.estimated_frames_label)

        # RUN BUTTON:
        # Starts the stratified selection process in a background thread.
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_stratified_selection)
        left_layout.addWidget(self.run_button)

        # PROGRESS BAR:
        # Integrated in the main window (not a dialog), visible only while processing.
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # CLUSTER SELECTION COMBOBOX:
        # After processing, we list available clusters. The user can pick one to filter frames.
        cluster_layout = QHBoxLayout()
        cluster_label = QLabel("Select Cluster:")
        self.cluster_combo = QComboBox()
        self.cluster_combo.currentIndexChanged.connect(self.on_cluster_changed)
        cluster_layout.addWidget(cluster_label)
        cluster_layout.addWidget(self.cluster_combo)
        left_layout.addLayout(cluster_layout)

        # RESULTS TABLE:
        # Shows the frames chosen by the stratified sampling.
        # Columns: Video name, Frame Index, Cluster
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["Video", "Frame Index", "Cluster"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setSortingEnabled(True)
        # When user selects a frame from the table, we update the image on the right panel.
        self.result_table.itemSelectionChanged.connect(self.on_frame_selection_changed)
        left_layout.addWidget(self.result_table)

        # RIGHT PANEL SETUP:
        # The right panel shows the currently selected frame as an image.
        right_layout = QVBoxLayout(right_widget)
        self.image_label = QLabel("No frame selected.")
        self.image_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.image_label)

        # INTERNAL DATA:
        # results: store the entire result dictionary from the logic.
        # all_rows: a flat list of tuples (video_name, frame_idx, cluster) for easy filtering and display.
        self.results = {}
        self.selected_cluster = None
        self.all_rows = []

        # Initially, no clusters or frames are displayed.
        self.cluster_combo.setEnabled(False)
        self.result_table.setEnabled(False)

        # Whenever the folder path changes, re-estimate frames.
        self.folder_line_edit.textChanged.connect(self.update_frame_estimation)

        # Set initial window size.
        self.resize(1200, 600)

    def select_folder(self):
        """
        Opens a directory selection dialog and updates the folder_line_edit if the user picks a folder.
        Also triggers a frame estimation update to reflect the newly chosen folder.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if folder:
            self.folder_line_edit.setText(folder)
            self.update_frame_estimation()

    def update_frame_estimation(self):
        """
        Checks how many frames can be extracted given the current video folder and frame_skip.
        Updates the estimated_frames_label accordingly.
        If folder is invalid, shows "N/A".
        """
        video_folder = self.folder_line_edit.text().strip()
        if not video_folder or not os.path.isdir(video_folder):
            self.estimated_frames_label.setText("Estimated frames: N/A")
            return

        frame_skip = self.frame_skip_spin.value()
        total_extracted = self.estimate_total_frames(video_folder, frame_skip)
        self.estimated_frames_label.setText(f"Estimated frames: {total_extracted}")

    def estimate_total_frames(self, video_folder, frame_skip):
        """
        Calculates how many frames would be extracted if we process all videos in the given folder
        using the specified frame_skip.
        This helps the user choose suitable parameters before running.
        """
        video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        total_count = 0
        for vid in video_files:
            video_path = os.path.join(video_folder, vid)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # If frame_count > 0, extracted frames = ((frame_count - 1) // frame_skip) + 1
            # This formula assumes we take frame 0, then every frame_skip-th frame.
            # If frame_count=0 (empty video), extracted=0.
            if frame_count > 0:
                extracted = ((frame_count - 1) // frame_skip) + 1
            else:
                extracted = 0
            total_count += extracted
        return total_count

    def run_stratified_selection(self):
        """
        Validates parameters and attempts to run stratified frame selection in a background thread.
        - Checks if we have a valid folder.
        - Ensures we can extract at least n frames.
        - Disables the UI while processing.
        - Shows the progress bar.
        - Creates and starts the Worker.
        """
        video_folder = self.folder_line_edit.text().strip()
        if not video_folder or not os.path.isdir(video_folder):
            QMessageBox.warning(self, "Warning", "Please select a valid video folder.")
            return

        n = self.n_spin.value()
        k = self.k_spin.value()
        frame_skip = self.frame_skip_spin.value()

        total_extracted = self.estimate_total_frames(video_folder, frame_skip)
        if total_extracted < n:
            # If we don't have enough frames to meet the requested 'n', warn the user.
            QMessageBox.warning(
                self, "Warning",
                f"You asked for {n} frames but with frame_skip={frame_skip}, only {total_extracted} frames "
                f"can be extracted. Please lower your frame_skip or reduce n."
            )
            return

        # Disable UI components during processing.
        self.run_button.setEnabled(False)
        self.cluster_combo.clear()
        self.cluster_combo.setEnabled(False)
        self.result_table.setRowCount(0)
        self.result_table.setEnabled(False)
        self.all_rows = []
        self.image_label.setText("No frame selected.")
        self.statusBar().showMessage("Processing...")

        # Show and reset progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Create a worker to do the heavy lifting in the background.
        worker = Worker(video_folder, n, k, frame_skip)
        # Connect signals from worker to UI slots:
        worker.signals.finished.connect(self.on_finished)
        worker.signals.error.connect(self.on_error)
        worker.signals.progress.connect(self.on_progress)
        self.current_worker = worker

        # Start the worker in the thread pool.
        self.thread_pool.start(worker)

    @Slot(int)
    def on_progress(self, value):
        """
        Called whenever the worker emits a progress signal.
        Updates the progress bar's current value.
        """
        self.progress_bar.setValue(value)

    @Slot(dict)
    def on_finished(self, results):
        """
        Called when the worker finishes successfully.
        - Re-enable UI components.
        - Hide the progress bar.
        - Display the results (frames) in the table.
        - Populate the cluster combo box.
        """
        self.run_button.setEnabled(True)
        self.statusBar().showMessage("Completed.")
        self.progress_bar.setVisible(False)

        self.results = results

        # Flatten results into a list of (video, frame_idx, cluster) for easier display.
        rows = []
        cluster_set = set()
        for vid, data in results.items():
            for idx, cluster in zip(data["indices"], data["clusters"]):
                rows.append((vid, idx, cluster))
                if cluster is not None:
                    cluster_set.add(cluster)

        self.all_rows = rows

        # Populate the cluster combo: "All" + each cluster number.
        self.cluster_combo.clear()
        self.cluster_combo.addItem("All")
        for c in sorted(cluster_set):
            self.cluster_combo.addItem(str(c))
        self.cluster_combo.setEnabled(True)

        # By default, show all frames.
        self.selected_cluster = None
        self.show_frames_for_cluster(None)
        self.result_table.setEnabled(True)

    @Slot(str)
    def on_error(self, error_message):
        """
        Called if the worker encounters an error.
        - Re-enable UI.
        - Hide the progress bar.
        - Show an error message box.
        """
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", error_message)
        self.statusBar().showMessage("Error encountered.")

    def on_cluster_changed(self, index):
        """
        Called when the user selects a new cluster from the combo box.
        Updates the table to show only frames from the chosen cluster.
        If "All" is chosen, show all frames.
        """
        if index == -1:
            return  # No valid selection
        selected_text = self.cluster_combo.currentText()
        if selected_text == "All":
            self.selected_cluster = None
        else:
            # Convert cluster label to int if possible
            try:
                self.selected_cluster = int(selected_text)
            except ValueError:
                self.selected_cluster = None

        self.show_frames_for_cluster(self.selected_cluster)

    def show_frames_for_cluster(self, cluster):
        """
        Filters the all_rows list based on the selected cluster (or shows all if cluster=None).
        Populates the result_table with the filtered frames.
        """
        filtered = []
        for vid, frame_idx, c in self.all_rows:
            if cluster is None or c == cluster:
                filtered.append((vid, frame_idx, c))

        # Temporarily disable sorting to populate table
        self.result_table.setSortingEnabled(False)
        self.result_table.setRowCount(len(filtered))
        for i, (vid, frame_idx, c) in enumerate(filtered):
            self.result_table.setItem(i, 0, QTableWidgetItem(str(vid)))
            self.result_table.setItem(i, 1, QTableWidgetItem(str(frame_idx)))
            self.result_table.setItem(i, 2, QTableWidgetItem(str(c) if c is not None else "N/A"))

        # Re-enable sorting and sort by Video column for consistency
        self.result_table.setSortingEnabled(True)
        self.result_table.sortItems(0, Qt.AscendingOrder)
        self.result_table.repaint()

        # Reset the image label since no specific frame is selected yet.
        self.image_label.setText("No frame selected.")

    def on_frame_selection_changed(self):
        """
        Called when the selection in the result_table changes.
        If a frame is selected, load and display it on the right panel.
        If none is selected, show "No frame selected."
        """
        selected_items = self.result_table.selectedItems()
        if not selected_items:
            self.image_label.setText("No frame selected.")
            return

        # The selected_items are the cells. The first selected cell gives us the row.
        row = selected_items[0].row()
        vid_item = self.result_table.item(row, 0)
        frame_idx_item = self.result_table.item(row, 1)
        if not vid_item or not frame_idx_item:
            self.image_label.setText("No frame selected.")
            return

        vid = vid_item.text()
        try:
            frame_idx = int(frame_idx_item.text())
        except ValueError:
            self.image_label.setText("No frame selected.")
            return

        video_folder = self.folder_line_edit.text().strip()
        video_path = os.path.join(video_folder, vid)
        frame = self.load_frame(video_path, frame_idx)
        if frame is None:
            QMessageBox.warning(self, "Warning", "Could not load frame.")
            self.image_label.setText("No frame selected.")
            return

        # Convert the frame from BGR (OpenCV format) to RGB (for Qt) and display it.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # Scale the image to fit the label while maintaining aspect ratio.
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def load_frame(self, video_path, frame_index):
        """
        Loads a single frame from the video at the given frame_index.
        Returns:
        - frame (np.ndarray, BGR): If successful
        - None if the frame could not be loaded.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        # Set the video position to the desired frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return frame
