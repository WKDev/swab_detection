import os
from PIL import Image

def augment_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files from the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    for image_file in image_files:
        # Open the image
        img_path = os.path.join(input_folder, image_file)
        img = Image.open(img_path)

        # Generate augmentations
        augmentations = [
            ("flip_horizontal", img.transpose(Image.FLIP_LEFT_RIGHT)),
            ("flip_vertical", img.transpose(Image.FLIP_TOP_BOTTOM)),
            ("flip_both", img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)),
            ("rotate_90", img.rotate(90)),
            ("rotate_180", img.rotate(180)),
            ("rotate_270", img.rotate(270)),
        ]

        # Save augmented images
        for prefix, augmented_img in augmentations:
            output_filename = f"{prefix}_{image_file}"
            output_path = os.path.join(output_folder, output_filename)
            augmented_img.save(output_path)

        print(f"Processed {image_file}")

    print("Augmentation complete!")

# Set input and output folders
input_folder = "images"
output_folder = "augmented_images"

# Run the augmentation
augment_images(input_folder, output_folder)