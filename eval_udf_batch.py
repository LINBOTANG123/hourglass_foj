#!/usr/bin/env python3
import argparse, os, json, math, csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
import k_diffusion as K


def load_image_as_tensor(path, size_hw):
    """PIL PNG -> torch.FloatTensor [1,3,H,W] in [0,1], resized/center-cropped."""
    H, W = size_hw
    tf = transforms.Compose([
        transforms.Resize(H, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(H),
        transforms.ToTensor(),  # -> [0,1]
    ])
    img = Image.open(path).convert("RGB")
    t = tf(img)[None, ...]  # (1,3,H,W)
    return t


@torch.no_grad()
def sample_udf_for_batch(model, sigma_min, sigma_max, steps, in_ch, cond_batch, device):
    """
    cond_batch: (B,3,H,W) float in [0,1]
    returns x0: (B,in_ch,H,W)
    """
    B, _, H, W = cond_batch.shape
    x = torch.randn([B, in_ch, H, W], device=device) * sigma_max
    sigmas = K.sampling.get_sigmas_karras(steps, sigma_min, sigma_max, rho=7., device=device)
    x0 = K.sampling.sample_dpmpp_2m_sde(
        model, x, sigmas, extra_args={"aug_cond": cond_batch},
        eta=0.0, solver_type="heun", disable=True
    )
    return x0


def main():
    p = argparse.ArgumentParser(description="Evaluate UDF model on a folder of PNGs.")
    p.add_argument("--config", required=True, help="Path to your JSON config.")
    p.add_argument("--ckpt",   required=True, help="Path to .pth or .safetensors checkpoint.")
    p.add_argument("--img-dir", required=True, help="Folder with input PNGs.")
    p.add_argument("--gt-dir",  required=True, help="Folder with ground-truth *_field.npy.")
    p.add_argument("--out-dir", required=True, help="Folder to save predicted *_pred.npy.")
    p.add_argument("--steps", type=int, default=50, help="Sampling steps.")
    p.add_argument("--batch-size", type=int, default=4, help="Eval batch size.")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    p.add_argument("--units", choices=["pixel", "normalized"], default="pixel",
                   help="Compute MSE in pixel units (recommended) or normalized units.")
    p.add_argument("--u-scale", type=float, default=64.0,
                   help="Scale used during training normalization (pixels per 1.0).")
    p.add_argument("--clip-max", type=float, default=64.0,
                   help="Optional clipping cap for distances in pixels (applied to GT and preds when computing MSE).")
    p.add_argument("--save-csv", action="store_true", help="Save per-image MSEs as CSV in out-dir.")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load config & build model
    # ------------------------------------------------------------------
    cfg = K.config.load_config(args.config)
    model_cfg = cfg["model"]
    in_ch = model_cfg["input_channels"]
    cond_ch = model_cfg.get("cond_channels", 3)
    size = model_cfg["input_size"]   # [H, W], square assumed in your setup
    assert cond_ch == 3, "This eval script expects RGB conditioning (cond_channels=3)."
    assert len(size) == 2 and size[0] == size[1], "This script assumes square inputs."

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.seed is not None:
        torch.manual_seed(args.seed)

    inner = K.config.make_model(cfg)
    # Wrap with denoiser wrapper (EDM / Karras loss wrapper)
    model = K.config.make_denoiser_wrapper(cfg)(inner)
    model.eval().to(device)

    # Load checkpoint weights
    ckpt_path = Path(args.ckpt)
    if ckpt_path.suffix == ".safetensors":
        import safetensors.torch as safetorch
        print(f"Loading (EMA) weights from {ckpt_path} ...")
        sd = safetorch.load_file(str(ckpt_path))
        inner.load_state_dict(sd)
    else:
        print(f"Loading (EMA) weights from {ckpt_path} ...")
        obj = torch.load(ckpt_path, map_location="cpu")
        # Prefer EMA if present
        if "model_ema" in obj:
            inner.load_state_dict(obj["model_ema"])
        elif "model" in obj:
            inner.load_state_dict(obj["model"])
        else:
            raise ValueError("Checkpoint missing 'model' / 'model_ema' keys.")
        del obj

    sigma_min = model_cfg["sigma_min"]
    sigma_max = model_cfg["sigma_max"]

    # ------------------------------------------------------------------
    # Collect files
    # ------------------------------------------------------------------
    img_dir = Path(args.img_dir)
    gt_dir  = Path(args.gt_dir)
    pngs = sorted([p for p in img_dir.glob("*.png")])
    if not pngs:
        raise FileNotFoundError(f"No .png files found in {img_dir}")

    print(f"Found {len(pngs)} PNGs in {img_dir}")
    print(f"Expecting GT fields in {gt_dir} as '<stem>_field.npy'")

    # optional per-image CSV
    csv_writer = None
    csv_fh = None
    if args.save_csv:
        csv_path = Path(args.out_dir) / "per_image_mse.csv"
        csv_fh = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["stem", "mse"])

    # ------------------------------------------------------------------
    # Inference loop (batched)
    # ------------------------------------------------------------------
    def chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    mse_sum = 0.0
    pix_sum = 0

    for batch_paths in tqdm(list(chunk(pngs, args.batch_size)), desc="Evaluating"):
        # Load conditioning images
        cond_list = []
        stems = []
        H, W = size
        for p in batch_paths:
            cond_list.append(load_image_as_tensor(p, (H, W)))
            stems.append(p.stem)
        cond_batch = torch.cat(cond_list, dim=0).to(device)  # (B,3,H,W)

        # Sample UDF predictions
        with torch.no_grad(), K.utils.eval_mode(model):
            x0 = sample_udf_for_batch(model, sigma_min, sigma_max, args.steps, in_ch, cond_batch, device)
            # x0: (B, in_ch, H, W)
            x0_np = x0.detach().cpu().numpy()  # (B,C,H,W)

        # Save predictions & compute MSE vs GT
        B = x0_np.shape[0]
        for b in range(B):
            pred = x0_np[b, 0]  # (H,W) predicted normalized or pixel depending on training
            # Save predicted field as (H,W,1)
            out_path = Path(args.out_dir) / f"{stems[b]}_pred.npy"
            np.save(out_path, pred[..., None].astype(np.float32))

            # Load ground-truth
            gt_path = gt_dir / f"{stems[b]}_field.npy"
            if not gt_path.exists():
                # Warn & skip MSE for this sample
                print(f"[WARN] GT not found for {stems[b]}: {gt_path}")
                continue
            gt = np.load(gt_path).astype(np.float32)  # (H,W,1) raw pixels (as generated)

            # squeeze to (H,W)
            if gt.ndim == 3 and gt.shape[-1] == 1:
                gt = gt[..., 0]

            # ----- Bring both to the same units -----
            if args.units == "pixel":
                # Convert model output to pixel units by u_scale
                pred_px = pred * float(args.u_scale)
                # Clip both to clip_max for fair comparison (the model was trained with clipped targets)
                if args.clip_max is not None:
                    c = float(args.clip_max)
                    pred_px = np.clip(pred_px, 0.0, c)
                    gt_px = np.clip(gt, 0.0, c)
                else:
                    gt_px = gt
                diff = pred_px - gt_px
            else:
                # normalized units: clip and divide GT by u_scale
                c = float(args.clip_max) if args.clip_max is not None else None
                if c is not None:
                    gt = np.clip(gt, 0.0, c)
                gt_norm = gt / float(args.u_scale)
                diff = pred - gt_norm

            # accumulate SSE and pixel count
            se = float(np.sum(diff**2))
            npx = diff.size
            mse_sum += se
            pix_sum += npx

            if csv_writer is not None:
                csv_writer.writerow([stems[b], se / npx])

    dataset_mse = mse_sum / max(1, pix_sum)
    unit_str = "pixels^2" if args.units == "pixel" else "(normalized units)^2"
    print(f"\nDataset MSE: {dataset_mse:.8f} [{unit_str}] over {pix_sum} pixels")

    if csv_fh is not None:
        csv_fh.close()
        print(f"Saved per-image MSEs to {Path(args.out_dir)/'per_image_mse.csv'}")


if __name__ == "__main__":
    main()
