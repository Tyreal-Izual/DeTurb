import matplotlib.pyplot as plt
import re

# Function to read log data from a file
def read_log_data(file_path):
    with open(file_path, 'r') as file:
        log_data = file.read()
    return log_data

# Path to the recording.log file
log_file_path = 'C:\\Users\\Zouzh\\Desktop\\training recording\\Original dynamic training\\36000\\recording.log'

# Reading log data from file
log_data = read_log_data(log_file_path)

# Extracting iters, Loss, PSNR, and SSIM from log entries
pattern = r"iters (\d+)/\d+ -Time:\d+\.\d+ -LR:\d+\.\d+ -Loss (\d+\.\d+) -PSNR: (\d+\.\d+) dB; SSIM: (\d+\.\d+)"
matches = re.findall(pattern, log_data)

# Converting the extracted data to a list of tuples (iters, Loss, PSNR, SSIM)
data = [(int(m[0]), float(m[1]), float(m[2]), float(m[3])) for m in matches]

# Unzipping the data for plotting
iters, losses, psnrs, ssims = zip(*data)

# Plotting
plt.figure(figsize=(14, 8))

# Loss plot
plt.subplot(3, 1, 1)
plt.plot(iters, losses, label='Loss', color='red')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.grid(True)

# PSNR plot
plt.subplot(3, 1, 2)
plt.plot(iters, psnrs, label='PSNR', color='blue')
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('PSNR over Iterations')
plt.grid(True)

# SSIM plot
plt.subplot(3, 1, 3)
plt.plot(iters, ssims, label='SSIM', color='green')
plt.xlabel('Iteration')
plt.ylabel('SSIM')
plt.title('SSIM over Iterations')
plt.grid(True)

plt.tight_layout()
plt.show()
