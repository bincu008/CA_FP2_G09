import os

# Configuration
group_id = "G09"
num_cores = 64
output_directory = f"core_instructions_{group_id}"

# Create a folder to keep things organized
os.makedirs(output_directory, exist_ok=True)

# The default instruction to write into every file
#dummy_instruction = "DRAM,WRITE,TOKEN_O,0,0,99999,0\n"
dummy_instruction = "# Unused core\n"
for i in range(num_cores):
    filepath = f"core_{i}_{group_id}.csv"
    # filepath = os.path.join(output_directory, filename)
    
    with open(filepath, "w") as f:
        # Write the dummy instruction
        f.write(dummy_instruction)
        if i == 63:
            # the last file is for dummy instructions, we can use it to verify the output of each model
            # expected output is 122 element, from model 0 to model 9 in order: 9 (model 0) + 13 (model 1) + 22 (model 2) + 16 (model 3) + 16 (model 4) + 16 (model 5) + 16 (model 6) + 16 (model 7) + 16 (model 8) + 16 (model 9)
            f.write("# Read 113 dummy elements from the input token just to populate SRAM Bank 0\n")
            f.write("DRAM,READ,TOKEN_I,0,0,0,113\n")
            f.write("# Write those 113 elements to the Output Tensor starting at offset 9\n")
            f.write("DRAM,WRITE,TOKEN_O,0,0,9,113\n")

print(f"Successfully generated {num_cores} files in the this folder.")