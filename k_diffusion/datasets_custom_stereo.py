from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

# datasets_custom_stereo.py  (unchanged imports)

class FoJStereoDataset(Dataset):
    def __init__(self, root, foj_path, left_img_path, right_img_path, disp_path,
                 size=128, image_glob="*.png", channels=(0,), u_scale=None, clip_max=None):
        self.root = Path(root); self.size = int(size)
        self.foj_dir = Path(foj_path); self.left_dir = Path(left_img_path)
        self.right_dir = Path(right_img_path); self.disp_dir = Path(disp_path)
        self.channels = tuple(channels); self.u_scale = u_scale; self.clip_max = clip_max

        self.img_tf = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.size), transforms.ToTensor(),
        ])

        left_paths = sorted(self.left_dir.glob(image_glob))
        items, missing = [], []
        for pL in left_paths:
            stem = pL.stem
            base = stem[:-5] if stem.endswith("_left") else stem
            pR = self.right_dir / f"{base}_right.png"
            pF = self.foj_dir   / f"{base}_field.npy"
            pD = self.disp_dir  / f"{base}_disp.npy"
            if pL.exists() and pR.exists() and pF.exists() and pD.exists():
                items.append((pL, pR, pF, pD))
            else:
                if not pL.exists(): missing.append(str(pL.resolve()))
                if not pR.exists(): missing.append(str(pR.resolve()))
                if not pF.exists(): missing.append(str(pF.resolve()))
                if not pD.exists(): missing.append(str(pD.resolve()))
        if missing:
            raise FileNotFoundError(f"Missing paired files (first few): {missing[:5]}")
        self.items = items

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        pL, pR, pF, pD = self.items[idx]

        # Stereo conditioning (6,H,W)
        L = self.img_tf(Image.open(pL).convert("RGB"))
        R = self.img_tf(Image.open(pR).convert("RGB"))
        cond6 = torch.cat([L, R], dim=0)

        # ----- UDF (channel 0 of FoJ) -----
        foj_np = np.load(pF)  # (H,W) or (H,W,C)
        if foj_np.ndim == 2: foj_np = foj_np[..., None]
        udf_idx = self.channels[0] if self.channels else 0
        if udf_idx >= foj_np.shape[2]:
            raise IndexError(f"FoJ has {foj_np.shape[2]} channels; asked for {udf_idx}.")
        udf_t = torch.from_numpy(foj_np[..., udf_idx])[None].float()  # (1,H,W)

        # ----- Disparity (pixels) -----
        disp_np = np.load(pD)
        if disp_np.ndim == 3 and disp_np.shape[-1] == 1: disp_np = disp_np[..., 0]
        disp_t = torch.from_numpy(disp_np)[None].float()             # (1,H,W)

        # Resize
        if udf_t.shape[-2:] != (self.size, self.size):
            udf_t  = F.interpolate(udf_t[None],  size=self.size, mode="bilinear", align_corners=False).squeeze(0)
        if disp_t.shape[-2:] != (self.size, self.size):
            disp_t = F.interpolate(disp_t[None], size=self.size, mode="bilinear", align_corners=False).squeeze(0)

        # Clip/scale UDF only
        if self.clip_max is not None: udf_t.clamp_(min=0.0, max=float(self.clip_max))
        if self.u_scale is not None:  udf_t = udf_t / float(self.u_scale)

        # Model input: [UDF_norm, DISP_px]
        x2 = torch.cat([udf_t, disp_t], dim=0)  # (2,H,W)

        aug_vec_9 = torch.zeros(9)
        return (x2, aug_vec_9, cond6, disp_t), torch.tensor(0)
