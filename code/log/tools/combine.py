import matplotlib.pyplot as plt
import re

# Function to read log data from a file
def read_log_data(file_path):
    with open(file_path, 'r') as file:
        log_data = file.read()
    return log_data

# Path to the recording.log file
log_file_path = 'C:\\Users\\Zouzh\\Desktop\\New folder\\DeformConv\\135000\\recording.log'

# Reading log data from file
log_data = read_log_data(log_file_path)

# Pattern for extracting Training and Validation data
training_pattern = r"INFO: Training: iters (\d+)/\d+ -Time:\d+\.\d+ -LR:\d+\.\d+ -Loss (\d+\.\d+) -PSNR: (\d+\.\d+) dB; SSIM: (\d+\.\d+)"
validation_pattern = r"INFO: Validation: Iters (\d+)/\d+ - Loss (\d+\.\d+) - PSNR: (\d+\.\d+) dB; SSIM: (\d+\.\d+)"

# Extracting Training and Validation data
training_matches = re.findall(training_pattern, log_data)
validation_matches = re.findall(validation_pattern, log_data)

# Converting the extracted data to a list of tuples (iters, Loss, PSNR, SSIM) for both Training and Validation
training_data = [(int(m[0]), float(m[1]), float(m[2]), float(m[3])) for m in training_matches]
validation_data = [(int(m[0]), float(m[1]), float(m[2]), float(m[3])) for m in validation_matches]

# Unzipping the data for plotting
training_iters, training_losses, training_psnrs, training_ssims = zip(*training_data)
validation_iters, validation_losses, validation_psnrs, validation_ssims = zip(*validation_data)

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(14, 18))

# Training Loss plot
axs[0, 0].plot(training_iters, training_losses, label='Training Loss', color='red')
axs[0, 0].set_xlabel('Iteration')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].set_title('Training Loss over Iterations')
axs[0, 0].grid(True)

# Validation Loss plot
axs[0, 1].plot(validation_iters, validation_losses, label='Validation Loss', color='purple')
axs[0, 1].set_xlabel('Iteration')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].set_title('Validation Loss over Iterations')
axs[0, 1].grid(True)

# Training PSNR plot
axs[1, 0].plot(training_iters, training_psnrs, label='Training PSNR', color='blue')
axs[1, 0].set_xlabel('Iteration')
axs[1, 0].set_ylabel('PSNR (dB)')
axs[1, 0].set_title('Training PSNR over Iterations')
axs[1, 0].grid(True)

# Validation PSNR plot
axs[1, 1].plot(validation_iters, validation_psnrs, label='Validation PSNR', color='orange')
axs[1, 1].set_xlabel('Iteration')
axs[1, 1].set_ylabel('PSNR (dB)')
axs[1, 1].set_title('Validation PSNR over Iterations')
axs[1, 1].grid(True)

# Training SSIM plot
axs[2, 0].plot(training_iters, training_ssims, label='Training SSIM', color='green')
axs[2, 0].set_xlabel('Iteration')
axs[2, 0].set_ylabel('SSIM')
axs[2, 0].set_title('Training SSIM over Iterations')
axs[2, 0].grid(True)

# Validation SSIM plot
axs[2, 1].plot(validation_iters, validation_ssims, label='Validation SSIM', color='cyan')
axs[2, 1].set_xlabel('Iteration')
axs[2, 1].set_ylabel('SSIM')
axs[2, 1].set_title('Validation SSIM over Iterations')
axs[2, 1].grid(True)

plt.tight_layout()
plt.show()
