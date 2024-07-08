import os
import cv2
import numpy as np
from deepface import DeepFace
from tqdm import tqdm
from datetime import datetime

# Set up the paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, 'omrisFullPhotos')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_base_folder = os.path.join(script_dir, f'omrisCroppedPhotos_{timestamp}')
reference_image_path = os.path.join(script_dir, 'reference_image.jpg')

# Create confidence-based folders
confidence_folders = {}
for i in range(0, 100, 10):
    folder_name = f"{i:02d}-{i + 10:02d}"
    folder_path = os.path.join(output_base_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    confidence_folders[i] = folder_path

# Pre-load the model to avoid multiple downloads
print("Initializing face recognition model...")
DeepFace.verify(img1_path=reference_image_path, img2_path=reference_image_path)
print("Model initialized.")

# Get the reference face embedding
reference_embedding = \
DeepFace.represent(img_path=reference_image_path, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]


def find_best_face(image_path):
    try:
        # Detect all faces in the image
        faces = DeepFace.extract_faces(img_path=image_path, enforce_detection=False)
        if not faces:
            return None, 0

        # Get embeddings for all detected faces
        embeddings = DeepFace.represent(img_path=image_path, model_name="VGG-Face", enforce_detection=False)

        # Calculate cosine similarity between reference and each detected face
        similarities = []
        for embedding in embeddings:
            similarity = np.dot(reference_embedding, embedding["embedding"]) / (
                        np.linalg.norm(reference_embedding) * np.linalg.norm(embedding["embedding"]))
            similarities.append(similarity)

        # Find the face with the highest similarity
        best_face_index = np.argmax(similarities)
        best_similarity = similarities[best_face_index]
        return faces[best_face_index], best_similarity

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, 0


def crop_face(image_path, output_path, confidence):
    best_face, similarity = find_best_face(image_path)
    if best_face is not None:
        img = cv2.imread(image_path)
        facial_area = best_face["facial_area"]

        # Handle different possible formats of facial_area
        if isinstance(facial_area, dict):
            x = facial_area.get('x', 0)
            y = facial_area.get('y', 0)
            w = facial_area.get('w', 0)
            h = facial_area.get('h', 0)
        elif isinstance(facial_area, (list, tuple)) and len(facial_area) == 4:
            x, y, w, h = facial_area
        else:
            print(f"Unexpected facial_area format in {image_path}")
            return

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
        print(f"Face cropped and saved: {output_path} (Confidence: {confidence:.2f})")
        return True
    else:
        print(f"No face found in {image_path}")
        return False


def process_images():
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_folder, image_file)
        _, similarity = find_best_face(input_path)

        # Convert similarity to confidence percentage
        confidence = similarity * 100

        # Determine which folder to save in
        folder_index = min(int(confidence // 10) * 10, 90)
        output_folder = confidence_folders[folder_index]

        output_path = os.path.join(output_folder, f"cropped_{image_file}")
        crop_face(input_path, output_path, confidence)


if __name__ == "__main__":
    print(f"Processing images from: {input_folder}")
    print(f"Saving cropped images to: {output_base_folder}")
    process_images()
    print("Processing complete!")
