"""Evaluate DeTurb tensors before video encoding with Alex LPIPS and GT-flow metrics."""
import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import tempfile
import cv2
import numpy as np
import torch
from deturb.evaluation import score_sequence, summarize, state_digest
from deturb.utils.flow_warp_metric import FarnebackGroundTruthMotion
from deturb.utils.inference import load_model, resolve_device
from deturb.utils.sequence_io import decode_cache, digest
from deturb.utils.image_io import image_bgr
from deturb.utils.experiment import write_json
from deturb.utils.paths import parse_path_args, resolve_path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--data-root',help='Override the selected split root, not its parent dataset root')
    p.add_argument('--split',default='test')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--max-videos',type=int,default=0,help='0: all; positive values label a partial evaluation')
    p.add_argument('--max-frames',type=int,default=0,help='0: all; positive values label a partial evaluation')
    p.add_argument('--patch-size',type=int,default=128)
    p.add_argument('--overlap',type=int,default=32)
    a=parse_path_args(p,argv)
    if a.max_videos<0 or a.max_frames<0 or a.max_frames==1:p.error('Need at least two frames for temporal metrics')
    if a.patch_size<32 or a.patch_size%32 or not 0<=a.overlap<a.patch_size:p.error('Invalid tiling')
    if Path(a.output).exists():raise FileExistsError('Choose a new result output: '+a.output)
    logging.basicConfig(level=logging.INFO)
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError('Install the metrics extra: pip install ".[metrics]"') from error
    if importlib.metadata.version('lpips')!='0.1.4':raise RuntimeError('This protocol requires lpips==0.1.4')
    manifest=json.loads(Path(a.manifest).read_text())
    static=manifest.get('data_layout')=='static_frames'
    samples=manifest['splits'][a.split]['samples']
    if a.max_videos:samples=samples[:a.max_videos]
    names=[s['name'] for s in samples]
    if not names or len(names)!=len(set(names)):raise ValueError('Empty or duplicate sample names')
    if any(Path(n).name!=n or n in ('.','..') for n in names):raise ValueError('Invalid sample name')
    if a.data_root:
        root=Path(a.data_root)
    else:
        default_root=manifest.get('split_roots',{}).get(a.split,manifest.get('data_root'))
        if not default_root:default_root=str(Path(manifest['dataset_root'])/a.split)
        root=Path(resolve_path(default_root,Path(a.manifest).parent))
    device=resolve_device(a.device)
    model,_=load_model(a.config,a.checkpoint,device)
    perceptual=lpips.LPIPS(net='alex',version='0.1').eval().to(device)
    identity={'checkpoint_sha256':digest(a.checkpoint),'manifest_sha256':digest(a.manifest),
              'lpips_state_sha256':state_digest(perceptual),'lpips':'0.1.4','backbone':'alex',
              'opencv':cv2.__version__,'torch':str(torch.__version__),'split':a.split,
              'max_videos':a.max_videos,'max_frames':a.max_frames,
              'patch_size':a.patch_size,'overlap':a.overlap,'data_root':str(root),
              'partial':bool(a.max_videos or a.max_frames),
              'protocol':'RGB float pre-encoding; native GT Farneback; FB(0.01,0.5); RGB L1',
              'primary':{'psnr':'video_equal','ssim':'video_equal','lpips':'video_equal',
                         'tlpips':'video_equal','warp_error':'valid_pixel_mean'}}
    signature=json.dumps(identity,sort_keys=True)
    rows=[];counts={}
    for sample in samples:
        name=sample['name'];meta=sample if static else sample['sources']['gt']
        count=min(meta['frames'],a.max_frames) if a.max_frames else meta['frames']
        if count<2:raise ValueError('Temporal evaluation requires at least two frames: '+name)
        counts[name]=count
        with tempfile.TemporaryDirectory(prefix='deturb-metrics-') as temp:
            if static:
                frame_names=sample['frame_names']
                if len(frame_names)!=sample['frames'] or len(set(frame_names))!=len(frame_names):
                    raise ValueError('Invalid static frame coverage')
                if any(Path(n).name!=n for n in frame_names):raise ValueError('Invalid frame name')
                shape=(meta['height'],meta['width'])
                source=np.stack([image_bgr(root/name/'turb'/n,shape) for n in frame_names[:count]])
                target=image_bgr(root/name/'gt.jpg',shape)
            else:
                if any(sample['sources']['turb'][k]!=meta[k] for k in ('frames','height','width')):
                    raise ValueError('GT/turb metadata differs: '+name)
                source=decode_cache(root/'turb'/name,Path(temp)/'source.npy',sample['sources']['turb'],count)
                target=decode_cache(root/'gt'/name,Path(temp)/'target.npy',meta,count)
            result=score_sequence(source,target,static,{'deturb':model},model.contract,perceptual,
                                  device,name,count,FarnebackGroundTruthMotion(),
                                  patch_size=a.patch_size,overlap=a.overlap)
            rows.append({'video':name,'signature':signature,'metrics':result})
            del source,target
        logging.info('Completed %s (%d/%d)',name,len(rows),len(samples))
    summary=summarize(rows,names,signature,['deturb'],counts)
    for row in rows:row.pop('signature')
    write_json(a.output,{'schema_version':1,'identity':identity,'summary':summary,'per_video':rows})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
