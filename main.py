import os
import cv2
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime

# Set up the paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, 'omrisFullPhotos')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(script_dir, f'omrisCroppedPhotos_{timestamp}')
reference_image_path = os.path.join(script_dir, 'reference_image.jpg')

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)


def crop_face(image_path, output_path):
    try:
        # Verify the face
        result = DeepFace.verify(img1_path=reference_image_path,
                                 img2_path=image_path,
                                 enforce_detection=False)

        if result['verified']:
            # If face is verified, crop and save
            img = cv2.imread(image_path)
            face = result['face2']
            x, y, w, h = face['x'], face['y'], face['w'], face['h']

            # Calculate square crop
            square_size = max(w, h)
            center_x = x + w // 2
            center_y = y + h // 2

            square_left = max(0, center_x - square_size // 2)
            square_top = max(0, center_y - square_size // 2)
            square_right = min(img.shape[1], square_left + square_size)
            square_bottom = min(img.shape[0], square_top + square_size)

            face_square = img[square_top:square_bottom, square_left:square_right]
            face_square = cv2.resize(face_square, (256, 256))

            cv2.imwrite(output_path, face_square)
        else:
            print(f"No matching face found in {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")


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
