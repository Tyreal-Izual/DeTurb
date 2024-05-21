import cv2
import torch
import argparse
import os
import time
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from model.TMT import TMT_MS

def split_to_patches(h, w, s):
    nh = h // s + 1
    nw = w // s + 1
    ol_h = int((nh * s - h) / (nh - 1))
    ol_w = int((nw * s - w) / (nw - 1))
    h_start = 0
    w_start = 0
    hpos = [h_start]
    wpos = [w_start]
    for i in range(1, nh):
        h_start = hpos[-1] + s - ol_h
        if h_start + s > h:
            h_start = h - s
        hpos.append(h_start)
    for i in range(1, nw):
        w_start = wpos[-1] + s - ol_w
        if w_start + s > w:
            w_start = w - s
        wpos.append(w_start)
    return hpos, wpos

def test_spatial_overlap(input_blk, model, patch_size):
    _, c, l, h, w = input_blk.shape
    hpos, wpos = split_to_patches(h, w, patch_size)
    out_spaces = torch.zeros_like(input_blk)
    out_masks = torch.zeros_like(input_blk)
    for hi in hpos:
        for wi in wpos:
            input_ = input_blk[..., hi:hi+patch_size, wi:wi+patch_size]
            output_ = model(input_)
            out_spaces[..., hi:hi+patch_size, wi:wi+patch_size].add_(output_)
            out_masks[..., hi:hi+patch_size, wi:wi+patch_size].add_(torch.ones_like(input_))
    return out_spaces / out_masks

def restore_PIL(tensor, b, fidx):
    img = tensor[b, fidx, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
    img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
    return img

def get_args():
    parser = argparse.ArgumentParser(description='Video inference with overlapping patches')
    parser.add_argument('--patch_size', '-ps', dest='patch_size', type=int, default=256, help='spatial patch size')
    parser.add_argument('--temp_patch', type=int, default=12, help='temporal patch size')
    parser.add_argument('--resize_ratio', type=float, default=1.0, help='spatial resize ratio for both w and h')
    parser.add_argument('--start_frame', type=float, default=0.0,
                        help='first frame to be processed, if < 1, it is ratio w.r.t. the entire video, if >1, it is absolute value')
    parser.add_argument('--total_frames', type=int, default=-1, help='number of total frames to be processed')
    parser.add_argument('--input_folder', type=str, default=None, help='path of input folder containing videos')
    parser.add_argument('--single_video', type=str, default=None, help='path of a single video file to process')
    parser.add_argument('--out_folder', type=str, default=None, help='path of output folder to save processed videos')
    parser.add_argument('--model_path', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--save_images', action='store_true', help='save results as images')
    parser.add_argument('--save_video', action='store_true', help='save results as video')
    parser.add_argument('--concatenate_input', action='store_true', help='concatenate input and output frames')
    return parser.parse_args()

def process_video(video_path, output_path, args, net):
    print(f'processing {video_path}')
    video = cv2.VideoCapture(video_path)

    h, w = int(video.get(4)), int(video.get(3))
    fps = int(video.get(5))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(args.start_frame * total_frames) if args.start_frame < 1 else int(args.start_frame)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if args.total_frames > 0:
        total_frames = args.total_frames

    hh, ww = (int(h * args.resize_ratio), int(w * args.resize_ratio)) if args.resize_ratio != 1 else (h, w)

    frames = [cv2.resize(video.read()[1], (ww, hh)) for _ in range(total_frames)]
    video.release()

    patch_size = args.patch_size

    for _ in range(2):
        inp_frames = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in frames]
        inp_frames = [TF.to_tensor(TF.resize(img, (hh, ww))) for img in inp_frames]

        input_ = torch.stack(inp_frames, dim=1).unsqueeze(0).cuda()
        if max(hh, ww) < patch_size:
            recovered = net(input_)
        else:
            recovered = test_spatial_overlap(input_, net, patch_size)

        recovered = recovered.permute(0, 2, 1, 3, 4)
        frames = [cv2.cvtColor(restore_PIL(recovered, 0, j), cv2.COLOR_RGB2BGR) for j in range(recovered.shape[2])]

    if args.save_video:
        output_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (ww, hh))
        for frame in frames:
            output_writer.write(frame)
        output_writer.release()

if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(0)

    net = TMT_MS(num_blocks=[2, 3, 3, 4],
                 heads=[1, 2, 4, 8],
                 num_refinement_blocks=2,
                 warp_mode='none',
                 n_frames=args.temp_patch,
                 att_type='shuffle').cuda().train()

    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
    for name, param in net.named_parameters():
        param.requires_grad = False

    if args.single_video:
        video_file = args.single_video
        out_path = os.path.join(args.out_folder, os.path.splitext(os.path.basename(video_file))[0] + '.mp4')
        process_video(video_file, out_path, args, net)
    else:
        video_files = [f for f in os.listdir(args.input_folder) if f.endswith(('.mp4', '.avi'))]
        for video_file in video_files:
            vpath = os.path.join(args.input_folder, video_file)
            out_path = os.path.join(args.out_folder, os.path.splitext(video_file)[0] + '.mp4')
            process_video(vpath, out_path, args, net)
