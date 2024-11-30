# Define input and output file paths
input_file = "data/custom/wrong_test.txt"  # Replace with the path to your input file
output_file = "data/custom/test.txt"  # Replace with the desired path for the output file

# Open the input file for reading and output file for writing
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Extract the filename from the input line
        filename = line.strip().split("\\")[-1]  # Extract the last part of the path
        # Construct the new line format
        new_line = f"data/custom/images/{filename}\n"
        # Write the new line to the output file
        outfile.write(new_line)

print(f"Reformatted lines have been written to '{output_file}'!")