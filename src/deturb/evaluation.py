"""Single-pass restoration metrics with user-standard temporal metrics."""
import hashlib
import logging
import math
import torch
from deturb.utils.sequence_io import input_rgb
from deturb.utils.inference import iter_restored_frames
from deturb.utils.metrics import batch_image_psnr_ssim
from deturb.utils.perceptual_video import lpips_distance
from deturb.utils.video_metric_accumulator import PerceptualAccumulator
LOGGER=logging.getLogger(__name__)

def state_digest(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.inference_mode()
def score_sequence(source, target, static, models, contract, perceptual, device, name, count, warper=None, *, patch_size=128, overlap=32):
    streams={arm:iter_restored_frames(iter(source),count,model,contract,device,patch_size=patch_size,overlap=overlap)
             for arm,model in models.items()}
    accum={arm:PerceptualAccumulator(perceptual,name+'/'+arm) for arm in models}
    pixel={arm:{'psnr_sum':0.,'ssim_sum':0.} for arm in models}
    previous_gt=None
    warp={arm:{'weighted_error':0.,'weighted_gt':0.,'weighted_delta':0.,'weighted_flow':0.,'valid_pixels':0,'pairs':0,'mask_ratio_sum':0.} for arm in models}
    if warper:warper.reset()
    LOGGER.info('tLPIPS video: %s, find %d frames. Adapter: %%08d.png contiguous output indices',name,count)
    for i in range(count):
        gt=input_rgb(target if static else target[i],device).unsqueeze(0)
        gt_distance=lpips_distance(perceptual,previous_gt,gt) if previous_gt is not None else None
        motion=warper.motion(previous_gt,gt) if warper and previous_gt is not None else None
        for arm,stream in streams.items():
            frame=next(stream)
            if frame.index!=i:raise ValueError('Temporal outputs are not contiguous')
            pred=frame.restored.unsqueeze(0).to(device).clamp(0,1)
            metrics=batch_image_psnr_ssim(pred,gt)
            pixel[arm]['psnr_sum']+=metrics.psnr_sum;pixel[arm]['ssim_sum']+=metrics.ssim_sum
            if motion is not None:
                value,delta=warper.score(accum[arm].previous[1],pred,motion)
                pixels=motion['valid_pixels'];w=warp[arm]
                w['weighted_error']+=value*pixels;w['weighted_gt']+=motion['gt_warp_error']*pixels
                w['weighted_delta']+=delta*pixels;w['weighted_flow']+=motion['flow_magnitude']*pixels
                w['valid_pixels']+=pixels;w['pairs']+=1;w['mask_ratio_sum']+=motion['mask_ratio']
            accum[arm].update(i,pred,gt,gt_distance)
        previous_gt=gt
    for stream in streams.values():
        if next(stream,None) is not None:raise ValueError('Extra restoration frames')
    result={arm:{**acc.finish(),'psnr':pixel[arm]['psnr_sum']/count,'ssim':pixel[arm]['ssim_sum']/count}
            for arm,acc in accum.items()}
    for arm in models:
        w=warp[arm]
        result[arm].update(warp_pairs=w['pairs'],warp_valid_pixels=w['valid_pixels'],
            warp_error=w['weighted_error']/w['valid_pixels'] if w['valid_pixels'] else None,
            gt_warp_error=w['weighted_gt']/w['valid_pixels'] if w['valid_pixels'] else None,
            warp_delta=w['weighted_delta']/w['valid_pixels'] if w['valid_pixels'] else None,
            flow_magnitude=w['weighted_flow']/w['valid_pixels'] if w['valid_pixels'] else None,
            mask_ratio=w['mask_ratio_sum']/w['pairs'] if w['pairs'] else None)
        # The metric functions return NaN for empty valid sets as specified.
        # Strict JSON records use null plus the explicit frame/pair status.
        for key,value in list(result[arm].items()):
            if isinstance(value,float) and not math.isfinite(value):result[arm][key]=None
    return result

def summarize(rows, names, signature, models, counts):
    if len(rows)!=len(names) or {r['video'] for r in rows}!=set(names):
        raise ValueError('Missing or duplicate videos')
    for row in rows:
        if row['signature']!=signature or set(row['metrics'])!=set(models):raise ValueError('Result identity differs')
        for m in row['metrics'].values():
            if m['frames']!=counts[row['video']] or m['lpips_frames']!=m['frames'] or m['valid_pairs']!=m['frames']-1:
                raise ValueError('Incomplete frame/pair coverage')
            if m['warp_pairs']!=m['frames']-1 or m['warp_valid_pixels']<=0:
                raise ValueError('Incomplete/empty warping coverage')
    result={}
    for arm in models:
        result[arm]={}
        for metric in ['psnr','ssim','lpips','tlpips','warp_error','gt_warp_error','warp_delta','mask_ratio','flow_magnitude']:
            values=[r['metrics'][arm][metric] for r in rows]
            if not all(v is not None and math.isfinite(v) for v in values):
                result[arm][metric]=None
            else:result[arm][metric]=math.fsum(values)/len(values)
    pixel_weighted={};frame_lpips={}
    for arm in models:
        values=[r['metrics'][arm] for r in rows]
        pixels=sum(m['warp_valid_pixels'] for m in values)
        pixel_weighted[arm]={'valid_pixels':pixels,**{
            key:math.fsum(m[key]*m['warp_valid_pixels'] for m in values)/pixels
            for key in ('warp_error','gt_warp_error','warp_delta','flow_magnitude')}}
        frame_lpips[arm]=math.fsum(m['lpips']*m['frames'] for m in values)/sum(m['frames'] for m in values)
    return {'videos':len(rows),'frames':sum(counts[n] for n in names),
            'pairs':sum(counts[n]-1 for n in names),'video_equal':result,
            'lpips_frame_weighted':frame_lpips,'warping_valid_pixel':pixel_weighted}
