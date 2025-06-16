#Import Statements
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define input and output folders
input_folder = "./Task1Images"
output_folder = "./Task1DestinationImages"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to Generate SharpnessMap
def generate_sharpness_map(image_path, output_folder):
    print(f"Processing image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to find the Image. Please check the path.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian filter to get sharpness
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_map = np.abs(laplacian)

    # Normalize the sharpness map to 0-255
    norm_map = cv2.normalize(sharpness_map, None, 0, 255, cv2.NORM_MINMAX)

    # Display original image and heatmap
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Sharpness Heatmap")
    plt.imshow(norm_map, cmap='jet')
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Save heatmap image in the destination folder
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_heatmap.jpg")
    heatmap_bgr = cv2.applyColorMap(norm_map.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(output_path, heatmap_bgr)
    print(f"Heatmap saved to: {output_path}")

# Process all images in Task1Images folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        generate_sharpness_map(image_path, output_folder)