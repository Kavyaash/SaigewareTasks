#Import Statements
import os
import cv2
from PIL import Image
from IPython.display import display

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define input and output folders
input_folder = "./Task2Images"
output_folder = "./Task2DestinationImages"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def detect_and_crop_face(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        print(f"Unable to read image: {input_path}")
        return

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"No faces detected in {input_path}")
        return

    # Select the largest detected face
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    cropped = image[y:y+h, x:x+w]

    # Save the cropped image
    cv2.imwrite(output_path, cropped)
    print(f"Cropped image saved to: {output_path}")

    # Display the cropped image
    display(Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)))

# Process all image files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"cropped_{filename}")
        detect_and_crop_face(input_path, output_path)
