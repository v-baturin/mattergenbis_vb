import torch
import time
import subprocess

def last_line_check(filename):
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            return lines and lines[-1].startswith("Everything")
    except FileNotFoundError:
        return False

def get_total_gpu_memory():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    # Returns a list of memory used for each GPU
    return [int(x) for x in result.decode().strip().split('\n')]


logfile = "log_multiple1.txt" 

gpu_mem_log = []

while not last_line_check(logfile):
    mems = get_total_gpu_memory()  # List of memory used for each GPU in MB
    current_time = time.time() 
    gpu_mem_log.append((current_time,mems))
    time.sleep(0.5)  # Sleep for 0,5 second before checking again

# Save to file
with open("log_gpu_mem.txt", "w") as f:
    for step, mems in gpu_mem_log:
        mems_str = ','.join(str(m) for m in mems)
        f.write(f"{step}\t{mems}\n")

# Print after run
for step, mems in gpu_mem_log:
    print(f"Step {step}: {mems:.2f} MB")