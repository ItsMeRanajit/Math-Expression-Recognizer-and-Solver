# %%
import os
import random
import shutil

# Paths to your dataset and the destination folder
dataset_path = "archive/data/extracted_images"
new_dataset_path = "archive/data/extracted_new_images2"

# Number of images to keep per class folder
num_images_to_keep = 2000

# Create the new dataset path if it doesn't exist
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

# Loop through each class folder
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    
    if os.path.isdir(class_path):
        # Create the class folder in the new dataset path
        new_class_path = os.path.join(new_dataset_path, class_folder)
        if not os.path.exists(new_class_path):
            os.makedirs(new_class_path)

        # Get all image files in the class folder
        all_images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        # If the number of images in the folder is greater than 1000, randomly select 1000 images
        if len(all_images) > num_images_to_keep:
            images_to_copy = random.sample(all_images, num_images_to_keep)
        else:
            images_to_copy = all_images  # Copy all images if there are fewer than 1000

        # Copy selected images to the new dataset folder
        for image in images_to_copy:
            src_image_path = os.path.join(class_path, image)
            dest_image_path = os.path.join(new_class_path, image)
            shutil.copy(src_image_path, dest_image_path)
            print(f"Copied: {src_image_path} -> {dest_image_path}")

print("Finished copying images.")



