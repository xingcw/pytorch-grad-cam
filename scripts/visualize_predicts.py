import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, SequentialSampler
from multimae.tools.load_multimae import load_model, predict
from multimae.utils.datasets import build_multimae_pretraining_dataset
from multimae.utils.plot_utils import get_semseg_metadata, plot_predictions

    
def main():
    seed = 1
    torch.manual_seed(seed) # change seed to resample new mask
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    model_name = "depth-semseg"
    model, args = load_model(model_name)
    
    # configure for detectron dataset (for prediection)
    flightmare_path = Path(os.environ["FLIGHTMARE_PATH"])
    multimae_path = flightmare_path.parent / "vision_backbones/MultiMAE"
    eval_data_path = multimae_path / "datasets/test/val"
    metadata = get_semseg_metadata(eval_data_path)
    
    args.eval_data_path = str(eval_data_path)
    dataset_val = build_multimae_pretraining_dataset(args, args.eval_data_path)
    sampler_val = SequentialSampler(dataset_val)
    data_loader_val = DataLoader(
        dataset_val, 
        sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    res_dir = flightmare_path.parent / "vis_attention/gradcam/results"
    video_path = os.path.join(res_dir, f"depth-semseg_predict_rgb" + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 15, (2 * 224, 224))
    
    step = 0
    show_img = True
    for data, _ in tqdm(data_loader_val):
        step += 1
        if step > 200:
            break
        preds, masks = predict(data, model_name, model, args, device="cuda", metadata=metadata)
        preds = {domain: pred.detach().cpu() for domain, pred in preds.items()}
        masks = {domain: mask.detach().cpu() for domain, mask in masks.items()}

        for domain in ["rgb", "depth", "semseg"]:
            if domain not in masks:
                masks[domain] = torch.ones_like(list(masks.values())[0])

        res = plot_predictions(data, preds, masks, metadata=metadata, show_img=show_img)
        
        if show_img:
            plt.show(block=False)
            plt.pause(3)
            plt.close()

        rgb_pred = res["rgb_pred"].numpy() * 255
        rgb_pred = rgb_pred.astype(np.uint8)
        rgb_gt = res["rgb_gt"].numpy() * 255
        rgb_gt = rgb_gt.astype(np.uint8)
        rgb_pred = cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2BGR)
        rgb_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2BGR)
        cv2.imshow("RGB Predict", rgb_pred)
        cv2.imshow("RGB GT", rgb_gt)
        cv2.waitKey(100)
        img = cv2.hconcat([rgb_gt, rgb_pred])
        video.write(img)
        
    video.release()
        
        
if __name__ == "__main__":
    main()