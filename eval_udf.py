#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import k_diffusion as K

def load_ckpt_and_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    inner = K.config.make_model(config)
    state = ckpt.get("model_ema") or ckpt["model"]
    inner.load_state_dict(state, strict=True)
    model = K.config.make_denoiser_wrapper(config)(inner)
    model.eval().to(device)
    return model, config

def load_cond_image(img_path, size_hw, device):
    H, W = size_hw
    tf = transforms.Compose([
        transforms.Resize(H, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(H),
        transforms.ToTensor(),  # [0,1]
    ])
    im = Image.open(img_path).convert("RGB")
    t = tf(im).unsqueeze(0).to(device, dtype=torch.float32)  # (1,3,H,W)
    return t

@torch.no_grad()
def sample_udf(model, model_cfg, cond_img, steps=50, seed=None, device="cuda"):
    g = torch.Generator(device=device) if seed is not None else None
    if g is not None:
        g.manual_seed(int(seed))

    H = W = int(model_cfg["input_size"][0])
    in_ch = int(model_cfg["input_channels"])
    sigma_min = float(model_cfg["sigma_min"])
    sigma_max = float(model_cfg["sigma_max"])

    x = torch.randn(1, in_ch, H, W, generator=g, device=device) * sigma_max
    sigmas = K.sampling.get_sigmas_karras(steps, sigma_min, sigma_max, rho=7., device=device)
    extra = {"aug_cond": cond_img}  # FoJCondTransformerV2 expects an image conditioning tensor

    x0 = K.sampling.sample_dpmpp_2m_sde(
        model, x, sigmas, extra_args=extra, eta=0.0, solver_type="heun", disable=True
    )
    return x0  # (1,C,H,W)

def save_udf_npy(field_tensor, out_path, scale_to_pixels=64.0, clamp_min=0.0):
    pred = field_tensor.squeeze(0).detach().cpu().numpy()  # (C,H,W)
    pred = np.transpose(pred, (1, 2, 0)).astype(np.float32)  # (H,W,C)
    if clamp_min is not None:
        pred = np.maximum(pred, float(clamp_min))
    if scale_to_pixels is not None:
        pred = pred * float(scale_to_pixels)
    np.save(out_path, pred)

def save_heatmap_png(udf_npy_path, out_png, viz_dmax=64.0, cmap="turbo"):
    D = np.load(udf_npy_path)
    if D.ndim == 3 and D.shape[-1] == 1:
        D = D[..., 0]
    D = D.astype(np.float32)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(D, cmap=cmap, vmin=0, vmax=(viz_dmax if viz_dmax else None))
    ax.set_title("Predicted UDF (pixels)")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("px")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Infer UDF from a PNG using a trained diffusion checkpoint.")
    ap.add_argument("--ckpt", required=True, help="Path to .pth checkpoint (with config + model_ema).")
    ap.add_argument("--image", required=True, help="Path to conditioning PNG image.")
    ap.add_argument("--out-npy", help="Output .npy path (HxWx1). Default: <image_stem>_udf.npy")
    ap.add_argument("--out-vis", help="Optional heatmap-only PNG path (no overlay).")
    ap.add_argument("--steps", type=int, default=100, help="Karras sampler steps.")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    ap.add_argument("--scale-to-pixels", type=float, default=64.0,
                    help="Multiply model output by this before saving (set to 64.0 if trained with u_scale=64).")
    ap.add_argument("--no-clamp", action="store_true", help="Do not clamp negatives to 0.")
    ap.add_argument("--viz-dmax", type=float, default=64.0, help="Colorbar max (px) for the heatmap.")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)

    model, cfg = load_ckpt_and_model(args.ckpt, device)
    cond = load_cond_image(args.image, (cfg["model"]["input_size"][0], cfg["model"]["input_size"][1]), device)

    field = sample_udf(model, cfg["model"], cond, steps=args.steps, seed=args.seed, device=device.type)

    img_path = Path(args.image)
    out_npy = Path(args.out_npy) if args.out_npy else img_path.with_name(img_path.stem + "_udf.npy")
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    save_udf_npy(
        field,
        str(out_npy),
        scale_to_pixels=args.scale_to_pixels,    # default 64.0 per your training
        clamp_min=(None if args.no_clamp else 0.0)
    )
    print(f"[OK] Saved UDF: {out_npy}")

    if args.out_vis:
        Path(args.out_vis).parent.mkdir(parents=True, exist_ok=True)
        save_heatmap_png(str(out_npy), args.out_vis, viz_dmax=args.viz_dmax)
        print(f"[OK] Saved heatmap: {args.out_vis}")

if __name__ == "__main__":
    main()
