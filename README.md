# Face Recognition and Cropping Script


## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/face-recognition-cropping.git
   cd face-recognition-cropping
   ```

2. Install the required packages:
   ```
   pip install opencv-python numpy deepface tqdm
   ```

3. Prepare your input data:
   - Place your input images in a folder named `omrisFullPhotos` in the script directory.
   - Add a reference image named `reference_image.jpg` to the script directory.

## Configuration

The script uses a `CONFIG` dictionary for easy customization:

- `input_folder`: Name of the folder containing input images (default: 'omrisFullPhotos')
- `reference_image`: Filename of the reference image (default: 'reference_image.jpg')
- `output_folder_prefix`: Prefix for the output folder (default: 'omrisCroppedPhotos')
- `confidence_ranges`: Range for confidence-based sorting (default: 0 to 90, step 10)
- `face_model`: DeepFace model to use (default: "VGG-Face")
- `output_size`: Size of cropped face images (default: 256x256)
- `enforce_detection`: Whether to enforce face detection (default: False)
- `batch_size`: Number of images to process in each batch (default: 50)

Modify these values in the script to suit your needs.

## Usage

Run the script with:

```
python face_recognition_cropping.py
```

The script will:
1. Set up output folders based on confidence ranges.
2. Initialize the face recognition model with the reference image.
3. Process images from the input folder in batches.
4. Detect faces, compare them to the reference, and crop the best match.
5. Save cropped faces to appropriate confidence-based folders.
6. Display progress and results.

## Output

Cropped face images are saved in folders named `omrisCroppedPhotos_YYYYMMDD_HHMMSS`, with subfolders for different confidence ranges (e.g., `00-10`, `10-20`, etc.).

## License
[MIT License](LICENSE)


## Author

Omri Fahn - [@omrifahn](https://github.com/omrifahn)

Project Link: https://github.com/omrifahn/FaceCrop
