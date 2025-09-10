from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class FoJDataset(Dataset):
    def __init__(self, root, img_path, foj_path, size=256, image_glob="*.png", transform=None, channels=(0, 1), u_scale=None, clip_max=None):
        self.root     = Path(root)
        self.size     = size
        self.transform = transform
        self.img_dir   = Path(img_path)
        self.field_dir = Path(foj_path)
        self.image_paths = sorted(self.img_dir.glob(image_glob))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
        self._to_tensor = transforms.ToTensor()
        # NEW: select channels
        self.channels = tuple(channels)
        self.u_scale  = u_scale           # ← NEW
        self.clip_max = clip_max          # ← NEW

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_pil  = Image.open(img_path).convert("RGB")

        img_tensor = self._to_tensor(img_pil)
        aug_vec_9   = torch.zeros(9)
        stem   = img_path.stem
        foj_np = np.load(self.field_dir / f"{stem}_field.npy")   # H×W×C
        foj_np = foj_np[:, :, list(self.channels)]               # ← NEW: keep only selected channels
        foj_t  = torch.from_numpy(foj_np).permute(2, 0, 1).float()

        # (optional) ensure size matches
        # if foj_t.shape[-2:] != (self.size, self.size):
        #     print("size mismatch")
        #     foj_t = F.interpolate(foj_t[None], size=self.size, mode="bilinear",
        #                           align_corners=False).squeeze(0)

        # ---- NEW: clip → scale -----------------------------------
        if self.clip_max is not None:
            foj_t.clamp_(min=0.0, max=float(self.clip_max))
        if self.u_scale is not None:
            foj_t = foj_t / float(self.u_scale)

        return (foj_t, aug_vec_9, img_tensor), torch.tensor(0)

    def __len__(self):
        return len(self.image_paths)