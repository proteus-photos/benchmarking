import subprocess
import random
import json
import re
import os
import sys

# Define the ranges for each hyperparameter
param_ranges = {
    # "w_coeff": (0, 2.0),
    # "i_coeff": (0, 2.0),
    # "o_coeff": (0, 2.0),
    # "min_margin": (0.1, 0.5),
    # "gamma": (0.05, 0.8),
    # "sharpness": (0.5, 5.0),
    "lr1": (0.5, 5),
    "lr2": (0.5, 3),
    "t0" : (5, 20)
}

# Number of random trials
num_trials = int(1e9)

# Function to extract validation loss from the script output
def extract_validation_loss(output):
    print(output)
    return float(output.split()[-1].strip().replace("RETURN_VALUE:", "").strip())

import random
import json
import re
import sys
import os
from datetime import datetime
import select


def run_script_with_live_output(cmd, output_file):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    
    stdout_output = []
    stderr_output = []
    
    with open(output_file, 'w') as f:
        while True:
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            ret = select.select(reads, [], [])

            for fd in ret[0]:
                if fd == process.stdout.fileno():
                    line = process.stdout.readline()
                    if line:
                        # print(line, end='')
                        f.write(f"STDOUT: {line}")
                        f.flush()
                        stdout_output.append(line)
                if fd == process.stderr.fileno():
                    line = process.stderr.readline()
                    if line:
                        print(line, end='', file=sys.stderr)
                        f.write(f"STDERR: {line}")
                        f.flush()
                        stderr_output.append(line)
            
            if process.poll() is not None:
                break
    
    return_code = process.wait()
    
    if return_code != 0:
        print(f"Child script exited with return code {return_code}")
    
    return ''.join(stderr_output)

os.makedirs("logs", exist_ok=True)

results = []
for trial in range(num_trials):
    params = {
        param: random.uniform(range_min, range_max)
        for param, (range_min, range_max) in param_ranges.items()
    }
    
    params["id"] = trial
    
    # Construct the command to run your script
    cmd = ["python", "train_anchor.py"]
    for param, value in params.items():
        cmd.extend([f"--{param}", str(value)])

    print(f"\nStarting Trial {trial + 1}/{num_trials}")
    print("Parameters:", json.dumps(params, indent=2))

    output_file = os.path.join("logs/", f"trial_{trial+1}_output.txt")
    
    output = run_script_with_live_output(cmd, output_file)
    
    val_loss = extract_validation_loss(output)
    
    results.append({"params": params, "val_loss": val_loss})
    
    print(f"Trial {trial + 1}/{num_trials} completed. Validation Loss: {val_loss}")

    # Sort results by validation loss
    results.sort(key=lambda x: x["val_loss"])

    # Save results to a JSON file
    with open("hyperparameter_tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

print("Hyperparameter tuning completed. Results saved to hyperparameter_tuning_results.json")
print("Best configuration:")
print(json.dumps(results[0], indent=2))