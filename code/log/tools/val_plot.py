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

# Read all log files
# df1 = read_log_file('C:\\Users\\Zouzh\\Desktop\\New folder\\or result\\result1.log')
df2 = read_log_file('C:\\Users\\Zouzh\\Desktop\\New folder\\or trained result\\result.log')
df3 = read_log_file('C:\\Users\\Zouzh\\Desktop\\New folder\\DeformConv\\135000\\result.log')  # Path to the third log file

# Set 'video' as index for all DataFrames
# df1 = df1.set_index('video')
df2 = df2.set_index('video')
df3 = df3.set_index('video')

# Find common videos among all DataFrames (optional)
# common_videos = df1.index.intersection(df2.index).intersection(df3.index)
# df1 = df1.loc[common_videos]
common_videos = df2.index.intersection(df3.index)

df2 = df2.loc[common_videos]
df3 = df3.loc[common_videos]

# Plot PSNR, SSIM, and LPIPS for all log files
metrics = ['psnr', 'ssim', 'lpips']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    # plt.plot(df1.index, df1[metric], label='Result 1', marker='o')
    plt.plot(df2.index, df2[metric], label='TMT', marker='x')
    plt.plot(df3.index, df3[metric], label='TMT_DC', marker='^')  # Plotting for the third log file
    plt.title(f'Comparison of {metric.upper()}')
    plt.xlabel('Video')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()
