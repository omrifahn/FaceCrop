import os
from mtcnn import MTCNN
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Set up the paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, 'omrisFullPhotos')
output_folder = os.path.join(script_dir, 'omrisCroppedPhotos')

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize the MTCNN detector
detector = MTCNN()


def crop_face(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_img)

    if results:
        x, y, w, h = results[0]['box']
        center_x = x + w // 2
        center_y = y + h // 2
        square_size = max(w, h)

        top = max(0, center_y - square_size // 2)
        left = max(0, center_x - square_size // 2)
        bottom = min(img.shape[0], top + square_size)
        right = min(img.shape[1], left + square_size)

        face_square = img[top:bottom, left:right]
        face_square = cv2.resize(face_square, (256, 256))  # Resize to a standard size
        cv2.imwrite(output_path, face_square)
    else:
        print(f"No face detected in {image_path}")


def process_images():
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def process_image(image_file):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"cropped_{image_file}")
        crop_face(input_path, output_path)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, image_files), total=len(image_files), desc="Processing images"))


if __name__ == "__main__":
    print(f"Processing images from: {input_folder}")
    print(f"Saving cropped images to: {output_folder}")
    process_images()
    print("Processing complete!")
