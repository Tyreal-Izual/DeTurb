"""Checkpoint discovery and device logging."""
import os
import datetime
import torch

def get_time(path):
    tstring = path.split('_')[-1]
    return datetime.datetime.strptime(tstring,'%m-%d-%Y-%H-%M-%S')

def find_latest_checkpoint(all_log_dir, run_name):
    run_name_list = [v for v in os.listdir(all_log_dir) if v.startswith(run_name)]
    run_name_list.sort(reverse=True, key=get_time)
    for p in run_name_list:
        ckpt_path = os.path.join(all_log_dir, p, 'checkpoints', 'latest.pth')
        if os.path.exists(ckpt_path):
            return ckpt_path
    return None

def get_cuda_info(logger):
    cudnn_version = torch.backends.cudnn.version()
    count = torch.cuda.device_count()

    logger.info(f'__CUDNN VERSION: {cudnn_version}\n'
                f'__Number CUDA Devices: {count}')

    for device_id in range(count):
        device_name = torch.cuda.get_device_name(device_id)
        memory = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)

        logger.info(f'__CUDA Device {device_id} Name: {device_name}\n'
                    f'__CUDA Device {device_id} Total Memory [GB]: {memory}')
