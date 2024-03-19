import matplotlib.pyplot as plt
import pandas as pd

# Function to read log file and extract data
def read_log_file(filepath):
    data = {'video': [], 'psnr': [], 'ssim': [], 'lpips': []}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.split(',')
            data['video'].append(parts[0].split(':')[1])
            data['psnr'].append(float(parts[1].split(':')[1]))
            data['ssim'].append(float(parts[2].split(':')[1]))
            data['lpips'].append(float(parts[3].split(':')[1]))
    return pd.DataFrame(data)

# Read both log files
df1 = read_log_file('C:\\Users\\Zouzh\\Desktop\\New folder\\or result\\result1.log')
df2 = read_log_file('C:\\Users\\Zouzh\\Desktop\\New folder\\or trained result\\result.log')

# Ensure videos are matched between the two datasets (optional, depends on your data)
df1 = df1.set_index('video')
df2 = df2.set_index('video')
common_videos = df1.index.intersection(df2.index)
df1 = df1.loc[common_videos]
df2 = df2.loc[common_videos]

# Plot PSNR, SSIM, and LPIPS for both log files
metrics = ['psnr', 'ssim', 'lpips']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    plt.plot(df1.index, df1[metric], label='Result 1', marker='o')
    plt.plot(df2.index, df2[metric], label='Result 2', marker='x')
    plt.title(f'Comparison of {metric.upper()}')
    plt.xlabel('Video')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()
