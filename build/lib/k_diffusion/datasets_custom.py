# myproject/foj_dataset.py

from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T

class FoJDataset(Dataset):
    """
    Expects this layout:
      root/
        imgs/
          one_object_0.png
          one_object_1.png
          ...
        fields/
          one_object_0_field.npy
          one_object_1_field.npy
          ...

    Returns:
      foj_t  Tensor[C, H, W]   ← your .npy (H×W×C) resized
      dummy  Tensor(0)         ← placeholder label
      img_t  Tensor[3, H, W]   ← conditioning RGB image
    """
    def __init__(self, root, size=256, image_glob="*.png"):
        self.root      = Path(root)
        self.img_dir   = self.root / "imgs"
        self.field_dir = self.root / "fields"

        self.image_paths = sorted(self.img_dir.glob(image_glob))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

        # transforms for the RGB image
        self.img_tf = T.Compose([
            T.ToTensor(),            # → [3,H,W] in [0,1]
            T.Resize(size),          
            T.CenterCrop(size),
        ])
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # — 1) load & transform conditioning image —
        img_path = self.image_paths[idx]
        img      = Image.open(img_path).convert("RGB")
        img_t    = self.img_tf(img)

        # — 2) build the FoJ filename & load the numpy array —
        stem     = img_path.stem                  # e.g. "one_object_0"
        foj_path = self.field_dir / f"{stem}_field.npy"
        if not foj_path.exists():
            raise FileNotFoundError(f"FoJ file not found:\n  {foj_path}")
        foj_arr = np.load(str(foj_path))          # shape: H×W×C

        # — 3) to tensor & resize FoJ map —
        #    (H,W,C) → (C,H,W)
        foj_t = torch.from_numpy(foj_arr).permute(2,0,1).float()
        #    resize with bilinear
        foj_t = F.interpolate(
            foj_t.unsqueeze(0),
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # — 4) dummy label placeholder —
        dummy = torch.tensor(0, dtype=torch.long)

        return foj_t, dummy, img_t
