# SaigewareTasks

# Requirements
- opencv-python
- numpy
- matplotlib
- pillow
- ipython

In this project, all required libraries are installed inside the venv environment, To run the Tasks PyCharm or Jupyter can be configured to use it as the interpreter.

##  Task 1: Image Sharpness Map Generator

**Objective**:  
Generate a sharpness heatmap of an image by identifying sharp (in-focus) vs. blurry regions using the Laplacian filter and visualize it.

### How it works:
- Converts image to grayscale.
- Applies Laplacian operator to detect intensity changes.
- Normalizes the output and creates a heatmap using matplotlib and OpenCV.

### Output:
- Heatmap is displayed using `matplotlib`.
- The output of heatmap image is saved with suffix _heatmap with prefix of file name in `Task1Images/` directory.

---

##  Task 2: Overlapping Object Cropping (Face Detection)

**Objective**:  
Detect and crop the primary object (a face) from images that may contain overlapping elements.

### How it works:
- Uses Haar Cascade classifier to detect faces.
- Selects the largest detected face.
- Crops and saves the face image.

### Output:
- Input Image is provided in Task2Images.
- Cropped face images saved to `Task2DestinationImages/`.

# About venv
venv is a virtual environment that isolates the dependencies used in this project from your system-wide Python packages. It helps prevent conflicts and ensures consistent behavior when running the project on different systems.
