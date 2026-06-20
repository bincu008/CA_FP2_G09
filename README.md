Read the MLP_Slicing_Strategy_Analysis.md to understand the slicing method. This repo is ready to run, just clone it then run docker commands to get the result. Below are the steps to generate all instruction files again.

Steps to run:
1. Clone the repo, place the whole folder in submission_example. The folder has been correctly configured to match the naming check.
2. Run the generate_all_instruction.py. It will sequentially exececute model{i}_instruction.py script. Each script will generate instruction files for the corresponding model.
3. Run the command to test:
<pre>
# run the below block to clean then simulate
docker compose run --rm --entrypoint bash sim -lc "cd /workspace/TA && bash 09_clean.sh"
docker compose run --rm --entrypoint bash sim -lc "cd /workspace/TA && python3 00_check_fp2_naming.py --submission-dir /workspace/FP2/submission_example"
docker compose run --rm --entrypoint bash sim -lc "cd /workspace/TA && SUBMISSION_DIR=/workspace/FP2/submission_example bash 03_Compile_Execution.sh"

# verify the output
docker compose run --rm --entrypoint bash sim -lc "cd /workspace/TA && python3 04_verify.py"

# calculate profit
docker compose run --rm --entrypoint bash sim -lc "cd /workspace/TA && SUBMISSION_DIR=/workspace/FP2/submission_example python3 05_profit_batch.py"
</pre>

The comparison script check output binary array and compare it to a golden binary array. The output array is expected to have 122 elements.
<pre>
9 (model 0) + 13 (model 1) + 22 (model 2) + 16 (model 3) + 16 (model 4) + 16 (model 5) + 16 (model 6) + 16 (model 7) + 16 (model 8) + 16 (model 9)
</pre>

It is stricly in order. So we can test each model individually by running the script for that particular model, then fill the other element with dummy values.

Steps to do:
1. Run dummy_instruction.py: this will generate all 64 dummy instruction file. However the last instruction file will read the input from DRAM then fill the output with dummy values.
2. Run the model{i}_instruction.py you want to test.
3. The output array must be 122 elements. So, other than model we want to test, we must fill the other places with dummy values
4. Run the verify steps are above to test.

Usually I test the models sequentially, starting from model 0. If I want to test model 1, then I run script model 0 (make sure the output is correct first), then run the script for model 1. In the last instruction file, I can easily fill the other 100 elements with dummy values.
<pre>
output array = [output model 0] + [output model 1] + ... +[output model 8] + [output model 9]
</pre>

Always keeps the missing elements on the left hand side. If I did not run script for model 0, then I need to add more instruction to fill dummy values for the first 9 elements. Below is an example:

<pre>
DRAM,READ,TOKEN_I,0,0,0,read_length
DRAM,WRITE,TOKEN_O,0,0,offset,write_length

# Example we run model0_instruction.py, which generates 9 elements. So, the other 113 instruction needs to be dummy.
DRAM,READ,TOKEN_I,0,0,0,113
DRAM,WRITE,TOKEN_O,0,0,9,113

# The verify script will print this
status    : FAIL
fail      : 113/122

# Meaning the first 9 elements are correct, the rest 113 elements are wrong.
</pre>
