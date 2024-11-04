import cv2
import numpy as np
import os

# Load an image using OpenCV with error handling
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")
    # Convert from BGR (OpenCV) to RGB (Pillow)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Save the processed image (overwrite original image)
def save_image(image, image_path):
    # Convert image from RGB (Pillow) back to BGR (OpenCV) before saving
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image_bgr)

# Resize image
def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

# Normalize image
def normalize_image(image):
    image = image.astype(np.float32) / 255.0
    return image

# Apply Gaussian Blur
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Perform data augmentation: Flip and Rotate
def augment_image(image):
    # Flip the image horizontally
    flipped_image = cv2.flip(image, 1)

    # Rotate the image by 180 degrees
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return flipped_image, rotated_image

# Process and overwrite all images in the folder
def process_images_from_folder(folder_path):
    # List all image files in the directory
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            image = load_image(image_path)
        except FileNotFoundError as e:
            print(e)
            continue

        # Preprocessing steps (choose the operations you want to apply)
        resized_image = resize_image(image, 224, 224)
        normalized_image = normalize_image(resized_image)
        blurred_image = apply_gaussian_blur(resized_image)
        flipped_image, rotated_image = augment_image(resized_image)

        # Overwrite the original image with the processed one (for example, saving the resized image)
        save_image(resized_image, image_path)  # Overwriting with resized image

        # You can also overwrite with other transformations, like blurred_image, etc.
        # save_image(blurred_image, image_path)
        # save_image(flipped_image, image_path)
        # save_image(rotated_image, image_path)

# Main function
def main():
    folder_path = 'C:/Users/Balaji/Documents/Project/New folder (2)/New folder/squamous cell carcinoma'  # Replace with your folder path
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    process_images_from_folder(folder_path)

if __name__ == '__main__':
    main()


