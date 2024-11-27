import torch

# Load the .pth file
checkpoint = torch.load("checkpoints/yolov3_ckpt_61.pth", map_location=torch.device('cpu'))

# Prepare the output text file
with open("output.txt", "w") as f:
    for key, value in checkpoint.items():
        f.write(f"Key: {key}\n")
        
        # Handle the content
        if isinstance(value, dict):  # If value is a dictionary
            for sub_key, sub_value in value.items():
                f.write(f"  {sub_key}: {str(sub_value)[:500]}...\n")  # Truncate long values
        else:  # Handle non-dictionary content
            f.write(f"Value: {str(value)[:500]}...\n")  # Truncate long values
        
        f.write("\n")

print("Checkpoint content has been written to output.txt")
