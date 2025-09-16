# foj_cond_transformer_v2.py

import torch
import torch.nn as nn
from .image_transformer_v2 import (
    ImageTransformerDenoiserModelV2,
    TokenMerge,
    downscale_pos,
)
from .axial_rope import make_axial_pos  # same import used by base class


class FoJCondTransformerV2(ImageTransformerDenoiserModelV2):
    """
    FoJ denoiser conditioned on an RGB image.

    Changes vs your previous version:
    - Keep a GLOBAL conditioning vector (what) via GAP -> mapping_cond (unchanged conceptually).
    - ALSO inject SPATIAL conditioning tokens (where) aligned to the FoJ patch grid.
    """

    def __init__(
        self,
        levels,
        mapping,
        in_channels,     # = C_foj
        out_channels,    # = C_foj
        patch_size,      # H==W; must divide input size
        num_classes=0,
        mapping_cond_dim=128,  # global cond vector dim
        cond_channels=3,       # RGB
        **kw
    ):
        super().__init__(levels, mapping, in_channels, out_channels,
                         patch_size, num_classes, mapping_cond_dim, **kw)

        # ------- 1) GLOBAL image encoder (keeps your old "what" vector) -------
        self.image_encoder_global = nn.Sequential(
            nn.Conv2d(cond_channels, mapping_cond_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mapping_cond_dim, mapping_cond_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> [B, D, 1, 1]
        )

        # ------- 2) SPATIAL image encoder (new "where" tokens, patch-aligned) -------
        # Produce per-pixel features with same channel width as the token width at level 0.
        width0 = levels[0].width
        self.cond_pre = nn.Sequential(
            nn.Conv2d(cond_channels, width0, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Convert per-pixel features into tokens on the SAME patch grid as FoJ tokens.
        self.cond_patch_in = TokenMerge(width0, width0, patch_size)

        # Optional learnable scale for the additive fusion.
        self.cond_gate = nn.Parameter(torch.tensor(1.0))

        # Safety: ensure mapping_cond is actually wired in the base class.
        assert self.mapping_cond_in_proj is not None, \
            "mapping_cond_dim=0 in config → set a positive value (e.g., 128) to enable global image conditioning."

    def forward(self, x, sigma, aug_cond=None, class_cond=None, mapping_cond=None):
        if aug_cond is None:
            raise ValueError("FoJCondTransformerV2 requires aug_cond=image tensor [B,3,H,W]")

        B = aug_cond.size(0)
        # ---------- GLOBAL vector (what) ----------
        img_feat = self.image_encoder_global(aug_cond).view(B, -1)  # [B, D]

        # ---------- PATCH the FoJ input ----------
        # (match the base class's first lines, but we interleave our cond tokens)
        x = x.movedim(-3, -1)      # [B, H, W, C_foj]
        x = self.patch_in(x)       # [B, H/P, W/P, width0]

        # ---------- SPATIAL cond tokens (where) ----------
        c = self.cond_pre(aug_cond)           # [B, width0, H, W]
        c = c.movedim(-3, -1)                 # [B, H, W, width0]
        cond_tokens = self.cond_patch_in(c)   # [B, H/P, W/P, width0]

        # Additive fusion (simple & effective). Swap to concat+Linear if you prefer.
        x = x + self.cond_gate.to(x.dtype) * cond_tokens

        # Positional grid on the token lattice
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)

        # ---------- Mapping network (time/aug/class/global image vec) ----------
        if self.class_emb is not None and class_cond is None:
            raise ValueError("class_cond must be specified if num_classes > 0")

        c_noise   = torch.log(sigma) / 4
        time_emb  = self.time_in_proj(self.time_emb(c_noise[..., None]))
        # We are NOT using the base "aug_cond" 9D augmentation vector here, keep zeros:
        aug_emb   = self.aug_in_proj(self.aug_emb(x.new_zeros([x.shape[0], 9])))
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        mapping_emb = self.mapping_cond_in_proj(img_feat)
        cond = self.mapping(time_emb + aug_emb + class_emb + mapping_emb)

        # ---------- Hourglass transformer (unchanged) ----------
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x); poses.append(pos)
            x = merge(x); pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond)

        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, pos, cond)

        # ---------- Unpatch (unchanged) ----------
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, -3)  # [B, C_foj, H, W]
        return x
