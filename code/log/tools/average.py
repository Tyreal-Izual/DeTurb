def calculate_averages(file_path):
    # Initialize variables to store the sum of metrics and count of lines processed
    sum_psnr = sum_ssim = sum_lpips = 0
    count = 0

    # Open and read through the file, extracting and summing the metrics
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(', ')
            psnr = float(parts[1].split(':')[1])
            ssim = float(parts[2].split(':')[1])
            lpips = float(parts[3].split(':')[1])

            sum_psnr += psnr
            sum_ssim += ssim
            sum_lpips += lpips
            count += 1

    # Calculate averages
    avg_psnr = sum_psnr / count
    avg_ssim = sum_ssim / count
    avg_lpips = sum_lpips / count

    return avg_psnr, avg_ssim, avg_lpips


# Example usage
if __name__ == "__main__":
    import sys
    # python calculate_metrics.py path/to/your/result.log
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Please provide a file path.")
    else:
        file_path = sys.argv[1]
        avg_psnr, avg_ssim, avg_lpips = calculate_averages(file_path)
        print(f"Average PSNR: {avg_psnr}")
        print(f"Average SSIM: {avg_ssim}")
        print(f"Average LPIPS: {avg_lpips}")
