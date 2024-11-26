import os
import random

# Paths to your image and annotation folders
image_dir = "data/custom/images/"
label_dir = "data/custom/labels/"

# Percentage split for training and validation
train_ratio = 0.8  # 80% training, 20% validation

# Output files for training and validation sets
train_file = "data/custom/train.txt"
valid_file = "data/custom/valid.txt"

# Get all image files
images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg", ".bmp"))]

# Shuffle images
random.shuffle(images)

# Split into training and validation
split_index = int(len(images) * train_ratio)
train_images = images[:split_index]
valid_images = images[split_index:]

# Write to train.txt and valid.txt
with open(train_file, "w") as train_f:
    train_f.write("\n".join(train_images))

with open(valid_file, "w") as valid_f:
    valid_f.write("\n".join(valid_images))

print(f"Training set: {len(train_images)} images")
print(f"Validation set: {len(valid_images)} images")
