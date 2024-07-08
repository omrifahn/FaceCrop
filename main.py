import os
import cv2
import numpy as np
from deepface import DeepFace
from tqdm import tqdm
from datetime import datetime

# Configuration parameters
CONFIG = {
    'input_folder': 'omrisFullPhotos',
    'reference_image': 'reference_image.jpg',
    'output_folder_prefix': 'omrisCroppedPhotos',
    'confidence_ranges': range(0, 100, 10),
    'face_model': "VGG-Face",
    'output_size': (256, 256),
    'enforce_detection': False
}

def setup_folders(script_dir, timestamp):
    output_base = os.path.join(script_dir, f"{CONFIG['output_folder_prefix']}_{timestamp}")
    confidence_folders = {
        i: os.path.join(output_base, f"{i:02d}-{i+10:02d}")
        for i in CONFIG['confidence_ranges']
    }
    for folder in confidence_folders.values():
        os.makedirs(folder, exist_ok=True)
    return output_base, confidence_folders

def initialize_model(reference_image_path):
    print("Initializing face recognition model...")
    DeepFace.verify(img1_path=reference_image_path, img2_path=reference_image_path)
    reference_embedding = DeepFace.represent(
        img_path=reference_image_path,
        model_name=CONFIG['face_model'],
        enforce_detection=CONFIG['enforce_detection']
    )[0]["embedding"]
    print("Model initialized.")
    return reference_embedding

def find_best_face(image_path, reference_embedding):
    try:
        faces = DeepFace.extract_faces(img_path=image_path, enforce_detection=CONFIG['enforce_detection'])
        if not faces:
            return None, 0
        embeddings = DeepFace.represent(img_path=image_path, model_name=CONFIG['face_model'], enforce_detection=CONFIG['enforce_detection'])
        similarities = [
            np.dot(reference_embedding, emb["embedding"]) / (np.linalg.norm(reference_embedding) * np.linalg.norm(emb["embedding"]))
            for emb in embeddings
        ]
        best_index = np.argmax(similarities)
        return faces[best_index], similarities[best_index]
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, 0

def crop_and_save_face(image_path, output_path, facial_area):
    img = cv2.imread(image_path)
    if isinstance(facial_area, dict):
        x, y, w, h = [facial_area.get(key, 0) for key in ('x', 'y', 'w', 'h')]
    elif isinstance(facial_area, (list, tuple)) and len(facial_area) == 4:
        x, y, w, h = facial_area
    else:
        print(f"Unexpected facial_area format in {image_path}")
        return False

    square_size = max(w, h)
    center_x, center_y = x + w // 2, y + h // 2
    left = max(0, center_x - square_size // 2)
    top = max(0, center_y - square_size // 2)
    right = min(img.shape[1], left + square_size)
    bottom = min(img.shape[0], top + square_size)

    face_square = img[top:bottom, left:right]
    face_square = cv2.resize(face_square, CONFIG['output_size'])
    cv2.imwrite(output_path, face_square)
    return True

def process_image(image_file, input_folder, confidence_folders, reference_embedding):
    input_path = os.path.join(input_folder, image_file)
    best_face, similarity = find_best_face(input_path, reference_embedding)
    if best_face is None:
        print(f"No face found in {input_path}")
        return

    confidence = similarity * 100
    folder_index = min(CONFIG['confidence_ranges'], key=lambda x: abs(x - confidence))
    output_folder = confidence_folders[folder_index]
    output_path = os.path.join(output_folder, f"cropped_{image_file}")

    if crop_and_save_face(input_path, output_path, best_face["facial_area"]):
        print(f"Face cropped and saved: {output_path} (Confidence: {confidence:.2f})")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, CONFIG['input_folder'])
    reference_image_path = os.path.join(script_dir, CONFIG['reference_image'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_base, confidence_folders = setup_folders(script_dir, timestamp)
    reference_embedding = initialize_model(reference_image_path)

    print(f"Processing images from: {input_folder}")
    print(f"Saving cropped images to: {output_base}")

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in tqdm(image_files, desc="Processing images"):
        process_image(image_file, input_folder, confidence_folders, reference_embedding)

    print("Processing complete!")

if __name__ == "__main__":
    main()