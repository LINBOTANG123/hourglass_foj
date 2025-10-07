# foj_cond_transformer_v2_stereo_dual.py
import torch
import torch.nn as nn
from .image_transformer_v2 import (
    ImageTransformerDenoiserModelV2,
    TokenMerge,
    downscale_pos,
)
from .axial_rope import make_axial_pos


class FoJCondTransformerV2StereoDual(ImageTransformerDenoiserModelV2):
    """
    Stereo FoJ denoiser with TWO independent image encoders (left & right).

    Inputs:
      x:        (B, 2, H, W)  where channel 0 = UDF (already normalized by dataset u_scale),
                              and channel 1 = disparity in PIXELS.
      aug_cond: (B, 6, H, W)  [Left RGB (3), Right RGB (3)]

    Outputs:
      (B, 2, H, W)  same channel order; returns UDF in dataset units, disparity in PIXELS.

    Internally:
      - Disparity is normalized by disp_norm for stable training and scaled back at the end.
      - UDF passes through unchanged (already ~O(1) from the dataset).
    """

    def __init__(
        self,
        levels,
        mapping,
        in_channels,      # expect 2
        out_channels,     # expect 2
        patch_size,       # int or (Ph, Pw); must be square & divide H/W
        num_classes=0,
        mapping_cond_dim=128,
        cond_channels=6,  # 3 (left) + 3 (right)
        disp_norm=64.0,   # internal normalization for disparity (pixels)
        **kw
    ):
        super().__init__(levels, mapping, in_channels, out_channels,
                         patch_size, num_classes, mapping_cond_dim, **kw)

        # --- Validate channels ---
        if in_channels != 2 or out_channels != 2:
            raise ValueError(f"in/out channels must be 2 (UDF, DISP). Got in={in_channels}, out={out_channels}.")
        if cond_channels != 6:
            raise ValueError(f"cond_channels must be 6 (L/R RGB). Got {cond_channels}.")
        self.cond_channels = cond_channels

        # --- Scales as buffers so they move with .to(device) and save in state dict ---
        self.register_buffer("in_scale",  torch.tensor([1.0, 1.0 / float(disp_norm)]))  # multiply inputs by this
        self.register_buffer("out_scale", torch.tensor([1.0, float(disp_norm)]))        # multiply outputs by this

        # --- Patch size handling ---
        if isinstance(patch_size, int):
            P = (patch_size, patch_size)
        else:
            assert isinstance(patch_size, (tuple, list)) and len(patch_size) == 2, \
                "patch_size must be int or (Ph, Pw)."
            assert patch_size[0] == patch_size[1], "Use square patch_size."
            P = (int(patch_size[0]), int(patch_size[1]))

        width0 = levels[0].width

        # ---------- GLOBAL encoders (independent L/R) ----------
        def make_global_enc():
            return nn.Sequential(
                nn.Conv2d(3, mapping_cond_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mapping_cond_dim, mapping_cond_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),  # -> [B, D, 1, 1]
            )
        self.image_encoder_global_L = make_global_enc()
        self.image_encoder_global_R = make_global_enc()
        self.global_fuse = nn.Linear(2 * mapping_cond_dim, mapping_cond_dim)

        # ---------- SPATIAL encoders (independent L/R), patch-aligned ----------
        def make_spatial_enc():
            return nn.Sequential(
                nn.Conv2d(3, width0, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.cond_pre_L = make_spatial_enc()
        self.cond_pre_R = make_spatial_enc()
        self.cond_patch_in_L = TokenMerge(width0, width0, P)
        self.cond_patch_in_R = TokenMerge(width0, width0, P)

        # Learnable gates for additive fusion
        self.cond_gate_L = nn.Parameter(torch.tensor(1.0))
        self.cond_gate_R = nn.Parameter(torch.tensor(1.0))
        
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.expects_image_aug_cond = True

        # Safety: ensure mapping_cond is wired in the base class
        assert self.mapping_cond_in_proj is not None, (
            "mapping_cond_dim=0 in config; set a positive value (e.g., 128)."
        )

    def forward(self, x, sigma, aug_cond=None, class_cond=None, mapping_cond=None):
        # ---- Input checks ----
        if aug_cond is None:
            raise ValueError("Require aug_cond (B,6,H,W) = [L_RGB(3), R_RGB(3)].")
        if not (x.ndim == 4 and x.shape[1] == self.in_channels):
            raise ValueError(f"FoJ input shape must be (B, {self.in_channels}, H, W). Got {tuple(x.shape)}.")
        if not (aug_cond.ndim == 4 and aug_cond.shape[1] == 6):
            raise ValueError(f"aug_cond must be (B, 6, H, W). Got {tuple(aug_cond.shape)}.")

        B = aug_cond.size(0)

        # ---------- Split conditioning into left/right ----------
        L_img, R_img = aug_cond[:, :3], aug_cond[:, 3:]   # (B,3,H,W) each

        # ---------- GLOBAL vectors (L/R), fuse ----------
        gL = self.image_encoder_global_L(L_img).view(B, -1)  # (B, D)
        gR = self.image_encoder_global_R(R_img).view(B, -1)  # (B, D)
        g_fused = self.global_fuse(torch.cat([gL, gR], dim=1))  # (B, D)

        # ---------- Normalize inputs channel-wise for internal stability ----------
        # x[0]=UDF (already ~O(1) from dataset), x[1]=DISP (pixels) -> scale to ~O(1)
        # NOTE: cast scales to x.dtype for AMP safety
        x = x * self.in_scale.view(1, -1, 1, 1).to(dtype=x.dtype)

        # ---------- Patch FoJ input to tokens ----------
        x = x.movedim(-3, -1)      # (B, H, W, C)
        x = self.patch_in(x)       # (B, H/P, W/P, width0)

        # ---------- SPATIAL cond tokens (L/R), fuse additively ----------
        tL = self.cond_patch_in_L(self.cond_pre_L(L_img).movedim(-3, -1))
        tR = self.cond_patch_in_R(self.cond_pre_R(R_img).movedim(-3, -1))
        x = x + self.cond_gate_L.to(dtype=x.dtype) * tL + self.cond_gate_R.to(dtype=x.dtype) * tR

        # ---------- Positional encodings ----------
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(
            x.shape[-3], x.shape[-2], 2
        )

        # ---------- Mapping net (time + aug + class + global image vec) ----------
        if self.class_emb is not None and class_cond is None:
            raise ValueError("class_cond must be specified if num_classes > 0")

        c_noise   = torch.log(sigma) / 4
        time_emb  = self.time_in_proj(self.time_emb(c_noise[..., None]))
        aug_emb   = self.aug_in_proj(self.aug_emb(x.new_zeros([x.shape[0], 9])))
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        cond      = self.mapping(time_emb + aug_emb + class_emb + self.mapping_cond_in_proj(g_fused))

        # ---------- Hourglass ----------
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x); poses.append(pos)
            x = merge(x); pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond)

        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, pos, cond)

        # ---------- Unpatch & restore original units ----------
        x = self.out_norm(x)
        out = self.patch_out(x).movedim(-1, -3)  # (B, 2, H, W) in normalized space

        # Undo internal disparity normalization; UDF passes through unchanged
        out = out * self.out_scale.view(1, -1, 1, 1).to(dtype=out.dtype)
        return out
