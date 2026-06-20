import subprocess
import os
import sys

def main():
    print("Starting generation for all 10 models...")
    
    for i in range(10):
        script_name = f"model{i}_instruction.py"
        
        if not os.path.exists(script_name):
            print(f"[ERROR] Could not find {script_name} in current directory.")
            continue
            
        print(f"Executing {script_name}...")
        try:
            result = subprocess.run(
                [sys.executable, script_name], 
                check=True, 
                capture_output=True, 
                text=True
            )
            print(f"  -> {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"  -> [FAILED] Error executing {script_name}:\n{e.stderr}")
            
    print("\nGeneration complete! All core CSV files are ready for simulation.")

if __name__ == "__main__":
    main()