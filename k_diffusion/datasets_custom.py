from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class FoJDataset(Dataset):
    def __init__(self, root, size=256, image_glob="*.png", transform=None):
        self.root     = Path(root)
        self.size     = size
        self.transform = transform
        self.img_dir   = self.root / "imgs"
        self.field_dir = self.root / "fields"
        self.image_paths = sorted(self.img_dir.glob(image_glob))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
        self._to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_pil  = Image.open(img_path).convert("RGB")

        if self.transform is None:
            # no augmentation: just PIL→Tensor + zero aug_vec_9
            img_tensor = self._to_tensor(img_pil)
            aug_vec_9   = torch.zeros(9)
        else:
            out = self.transform(img_pil)
            if isinstance(out, tuple):
                if len(out) == 2:
                    img_tensor, aug_vec_9 = out
                elif len(out) == 3:
                    img_tensor, aug_vec_9, _ = out
                else:
                    raise ValueError(f"Unexpected transform output length: {len(out)}")
            else:
                img_tensor = out
                aug_vec_9   = torch.zeros(9)

        stem   = img_path.stem
        foj_np = np.load(self.field_dir / f"{stem}_field.npy")   # H×W×C
        foj_t = torch.from_numpy(foj_np).permute(2, 0, 1).float()

        # return ((foj, aug_vec_9, img), dummy_label)
        return (foj_t, aug_vec_9, img_tensor), torch.tensor(0)

    def __len__(self):
        return len(self.image_paths)
