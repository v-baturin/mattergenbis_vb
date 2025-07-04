import torch
import time

def last_line_check(filename):
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            return lines and lines[-1].startswith("Everything")
    except FileNotFoundError:
        return False

logfile = "log_multiple1.txt" 

gpu_mem_log = []

while not last_line_check(logfile):
    mem = torch.cuda.memory_allocated() / 1024**2  # in MB
    current_time = time.time() 
    gpu_mem_log.append((current_time,mem))
    time.sleep(0.5)  # Sleep for 1 second before checking again

# Save to file
with open("log_gpu_mem.txt", "w") as f:
    for step, mem in gpu_mem_log:
        f.write(f"{step}\t{mem}\n")

# Print after run
for step, mem in gpu_mem_log:
    print(f"Step {step}: {mem:.2f} MB")