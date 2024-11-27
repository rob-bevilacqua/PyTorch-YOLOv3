from PIL import Image
import os

# Define the directory containing the .bmp images
input_folder = "data/custom/images/"  # Replace with your folder path
output_folder = "data/custom/jpgImages/"  # Replace with output folder path (or same as input)

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".bmp"):  # Check if the file is a .bmp image
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")  # Keep the same name

        # Open the .bmp image
        with Image.open(input_path) as img:
            # Convert and save as .jpg
            img.convert("RGB").save(output_path, "JPEG")
        print(f"Converted {filename} to {os.path.basename(output_path)}")
