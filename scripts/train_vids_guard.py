#!/usr/bin/env python3
import argparse, os, sys, json, time, random, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import csv, os


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# plotting + metrics (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
HAS_SK = True  # if you already have this, keep only one




# Optional libs
try:
    import timm
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False
    warnings.warn("timm not found; will fallback to torchvision.resnet18")

try:
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from sklearn.metrics import roc_auc_score
    HAS_SK = True
except Exception:
    HAS_SK = False
    warnings.warn("sklearn not found; AUC will be reported as NaN")

from PIL import Image, ImageFilter, ImageEnhance
import cv2


IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp'}
VID_EXTS = {'.mp4','.avi','.mov','.mkv','.webm'}

# --- Robust video probe (NEW) ---
def probe_video_ok(video_path: Path, min_frames: int = 4) -> bool:
    """
    Return True if the video looks decodable: file exists, not tiny, can open,
    has >= min_frames, and we can read at least one frame.
    """
    try:
        if not video_path.exists() or video_path.stat().st_size < 1024:
            return False
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return False
        return n >= min_frames
    except Exception:
        return False

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def is_video(p: Path) -> bool:
    return p.suffix.lower() in VID_EXTS

def decode_video_total_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

def read_video_frame(video_path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# --- Discover dataset with broken-file logging (REPLACE) ---
def discover_dataset(root: Path, broken_log: Optional[Path] = None) -> List[Tuple[str,int,List[Path]]]:
    """
    Accepts either:
      A) root/{real,fake}/<video-id>/*.{jpg,png,...}   (folders of frames)
      B) root/{real,fake}/*.{mp4,avi,mov,mkv,webm}     (video files)
    Returns tuples (video_id, label_int, sources)
      - For frame folders: sources = list of frame Paths
      - For videos:        sources = [single Path to video file]
    Skips obviously broken videos and optionally logs them to broken_log.
    """
    items: List[Tuple[str,int,List[Path]]] = []
    bads: List[str] = []

    for label_name, label_int in [('real',0), ('fake',1)]:
        label_dir = root / label_name
        if not label_dir.exists():
            warnings.warn(f"Missing folder: {label_dir}")
            continue

        # Case B: video files directly under {real,fake}
        vids = [p for p in sorted(label_dir.iterdir()) if p.is_file() and is_video(p)]
        for vp in vids:
            if probe_video_ok(vp):
                vid_id = f"{label_name}/{vp.stem}"
                items.append((vid_id, label_int, [vp]))
            else:
                bads.append(str(vp))

        # Case A: folders of frames under {real,fake}
        for vd in sorted([d for d in label_dir.iterdir() if d.is_dir()]):
            frames = sorted([p for p in vd.iterdir() if p.is_file() and is_image(p)])
            if frames:
                items.append((f"{label_name}/{vd.name}", label_int, frames))

    if broken_log is not None and bads:
        broken_log.parent.mkdir(parents=True, exist_ok=True)
        with open(broken_log, "w") as f:
            f.write("\n".join(bads) + "\n")
        print(f"[WARN] Skipped {len(bads)} broken videos. List saved to: {broken_log}")

    if len(items) == 0:
        print(f"[ERROR] No valid data found under {root}/{{real,fake}}", file=sys.stderr)
    else:
        print(f"[INFO] Discovered {len(items)} items. Skipped broken videos: {len(bads)}")

    return items

# ===== VIDS-Guard custom backbone (VGBBackbone) =====

class BlurPool(nn.Module):
    def __init__(self, channels, filt_size=3, stride=2):
        super().__init__()
        assert filt_size in (3, 5)
        if filt_size == 3:
            kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
        else:
            kernel = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
        filt = kernel[:, None] * kernel[None, :]
        filt = filt / filt.sum()
        self.register_buffer("filt", filt[None, None, :, :].repeat(channels, 1, 1, 1))
        self.stride = stride
        self.groups = channels
        self.pad = filt_size // 2

    def forward(self, x):
        return F.conv2d(x, self.filt, stride=self.stride, padding=self.pad, groups=self.groups)

class DSResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.dw1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.dw2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.pw2 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, max(8, out_ch // 8), 1), nn.SiLU(inplace=True),
            nn.Conv2d(max(8, out_ch // 8), out_ch, 1), nn.Sigmoid()
        ) if use_se else nn.Identity()
        self.down = None
        self.blur = None
        if stride == 2:
            self.blur = BlurPool(in_ch, filt_size=3, stride=2)
            if in_ch != out_ch:
                self.down = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
        elif in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
    def forward(self, x):
        identity = x
        if self.blur is not None:
            x = self.blur(x)
            if identity.shape[-1] != x.shape[-1] or identity.shape[-2] != x.shape[-2]:
                identity = F.interpolate(identity, size=x.shape[-2:], mode="nearest")
        out = self.act(self.bn1(self.pw1(self.dw1(x))))
        out = self.bn2(self.pw2(self.dw2(out)))
        if isinstance(self.se, nn.Sequential):
            out = out * self.se(out)
        if self.down is not None:
            identity = self.down(identity)
        out = self.act(out + identity)
        return out

def srm_kernels():
    k1 = torch.tensor([[0, 0, 0],[0, 1, -1],[0, 0, 0]], dtype=torch.float32)
    k2 = torch.tensor([[0, 0, 0],[0, 1, 0],[0, -1, 0]], dtype=torch.float32)
    k3 = torch.tensor([[0, 1, 0],[-1, 0, 1],[0, -1, 0]], dtype=torch.float32)
    K = torch.stack([k1, k2, k3], dim=0)[:, None, :, :]
    K = K / (K.abs().sum(dim=(2, 3), keepdim=True) + 1e-6)
    return K

def rgb_to_ycbcr_weight():
    M = torch.tensor([[ 0.2990,  0.5870,  0.1140],
                      [-0.1687, -0.3313,  0.5000],
                      [ 0.5000, -0.4187, -0.0813]], dtype=torch.float32)
    b = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32)
    return M, b


class SRMBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # depthwise 3x3, one filter per channel
        self.srm = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.srm.weight.requires_grad_(False)
        self.srm.weight.data = srm_kernels()  # shape [3,1,3,3]
        self.bn = nn.BatchNorm2d(3)
        self.act = nn.SiLU(inplace=True)

        self.stage1 = nn.Sequential(DSResBlock(3, 48, stride=1), DSResBlock(48, 48, stride=2))
        self.stage2 = nn.Sequential(DSResBlock(48, 96, stride=1), DSResBlock(96, 96, stride=2))
        self.stage3 = nn.Sequential(DSResBlock(96,192, stride=1), DSResBlock(192,192, stride=2))
        self.head   = nn.Conv2d(192, 256, 1, bias=False)
        self.hbn    = nn.BatchNorm2d(256)
    def forward(self, x):
        x = self.act(self.bn(self.srm(x)))
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        x = self.hbn(self.head(x))
        return x



class DCTProxyBranch(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.pool8 = nn.AvgPool2d(8, 8)
        self.conv1 = nn.Conv2d(in_ch, 32, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.act   = nn.SiLU(inplace=True)
        self.block = nn.Sequential(
            DSResBlock(32, 64, stride=1),
            DSResBlock(64, 64, stride=1),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        x = self.pool8(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.block(x)
        return x  # (B,128,~H/8,~W/8)

class ColorBranch(nn.Module):
    def __init__(self):
        super().__init__()
        M, b = rgb_to_ycbcr_weight()
        self.register_buffer("M", M)
        self.register_buffer("b", b.view(1, 3, 1, 1))
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
        )
        self.stage1 = DSResBlock(32, 32, stride=2)
        self.stage2 = DSResBlock(32, 64, stride=2)
        self.stage3 = DSResBlock(64, 128, stride=2)

    def forward(self, x):
        if x.ndim != 4:
            raise RuntimeError(f"ColorBranch expects 4D input, got {x.shape}")
        if x.shape[1] == 3:
            x_chw = x
        elif x.shape[-1] == 3:
            x_chw = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(f"Expected 3 channels, got shape {x.shape}")
        ycbcr = torch.einsum('bchw,dc->bdhw', x_chw, self.M) + self.b
        ycbcr = ycbcr.clamp(0, 1)
        z = self.stem(ycbcr)
        z = self.stage1(z); z = self.stage2(z); z = self.stage3(z)
        return z  # (N,128,H/8,W/8)

class VGBBackbone(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.branch_srm   = SRMBranch()
        self.branch_freq  = DCTProxyBranch()
        self.branch_color = ColorBranch()
        self.fuse = nn.Sequential(
            nn.Conv2d(256 + 128 + 128, 512, 1, bias=False),
            nn.BatchNorm2d(512), nn.SiLU(inplace=True),
        )
        self.proj = nn.Linear(512, out_dim) if out_dim != 512 else nn.Identity()
        self.out_dim = out_dim
    def forward(self, x):
        a = self.branch_srm(x)
        b = self.branch_freq(x)
        c = self.branch_color(x)
        if b.shape[-2:] != a.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        if c.shape[-2:] != a.shape[-2:]:
            c = F.interpolate(c, size=a.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([a, b, c], dim=1)
        x = self.fuse(x)
        x = x.mean(dim=(2, 3))
        x = self.proj(x)
        return x  # (B,out_dim)
# ===== end VGBBackbone =====

# ------------------- Utils -------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def split_items(items, train_ratio=0.8, val_ratio=0.1, seed=41):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(items))
    n_train = int(len(perm)*train_ratio)
    n_val   = int(len(perm)*val_ratio)
    idx_tr = set(perm[:n_train])
    idx_v  = set(perm[n_train:n_train+n_val])
    idx_te = set(perm[n_train+n_val:])
    train = [items[i] for i in range(len(items)) if i in idx_tr]
    val   = [items[i] for i in range(len(items)) if i in idx_v]
    test  = [items[i] for i in range(len(items)) if i in idx_te]
    return train, val, test

# ------------------- Augmentations -------------------

def random_jpeg(img: Image.Image, q_low=30, q_high=90) -> Image.Image:
    q = random.randint(q_low, q_high)
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def random_blur(img: Image.Image, p=0.2) -> Image.Image:
    if random.random() < p:
        r = 0.5 + random.random()
        return img.filter(ImageFilter.GaussianBlur(radius=r))
    return img

def random_bc(img: Image.Image, strength=0.2) -> Image.Image:
    if strength <= 0: return img
    b = 1.0 + random.uniform(-strength, strength)
    c = 1.0 + random.uniform(-strength, strength)
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    return img

class ImgTransforms:
    def __init__(self, img_size=224, train=True, jpeg_q=(30,90), blur_p=0.2, bc_strength=0.2,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.img_size = img_size
        self.train = train
        self.jpeg_q = jpeg_q
        self.blur_p = blur_p
        self.bc_strength = bc_strength
        # keep arrays on the instance so workers don't rely on module globals
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if self.train:
            w, h = img.size
            scale = random.uniform(0.7, 1.0)
            nw, nh = int(w*scale), int(h*scale)
            if nw>0 and nh>0:
                x1 = random.randint(0, max(0, w-nw))
                y1 = random.randint(0, max(0, h-nh))
                img = img.crop((x1,y1,x1+nw,y1+nh))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = random_bc(img, self.bc_strength)
            img = random_blur(img, self.blur_p)
            img = random_jpeg(img, self.jpeg_q[0], self.jpeg_q[1])

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2,0,1).astype(np.float32)
        return torch.from_numpy(arr)


# ------------------- Dataset -------------------

class ClipDataset(Dataset):
    def __init__(self, items, split: str, T: int, img_size: int,
                 clips_per_video=1, frame_drop_p=0.1, temporal_jitter=1,
                 jpeg_q=(30,90), blur_p=0.2, bc_strength=0.2):
        assert split in {'train','val','test'}
        self.base_items = items
        self.split = split
        self.T = T
        self.frame_drop_p = frame_drop_p
        self.temporal_jitter = temporal_jitter

        self.tf_train = ImgTransforms(img_size, True, jpeg_q, blur_p, bc_strength)
        self.tf_eval  = ImgTransforms(img_size, False)

        if split == 'train' and clips_per_video > 1:
            self.items = []
            for it in items:
                self.items.extend([it] * clips_per_video)
        else:
            self.items = items

    def __len__(self):
        return len(self.items)

    def _load_frame(self, path: Path, train: bool) -> torch.Tensor:
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(f"Cannot read: {path}")
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        tfm = self.tf_train if train else self.tf_eval
        return tfm(img)

    def _choose_indices(self, n: int, train: bool) -> List[int]:
        if n <= self.T:
            idxs = list(range(n)) + [n-1]*(self.T-n)
            return idxs
        step = n / self.T
        idxs = [int(i*step) for i in range(self.T)]
        if train and self.temporal_jitter > 0:
            idxs = [min(max(0, j + random.randint(-self.temporal_jitter, self.temporal_jitter)), n-1) for j in idxs]
            if random.random() < self.frame_drop_p:
                k = random.randrange(1, self.T)
                idxs[k] = idxs[k-1]
        return idxs


    def __getitem__(self, i):
        for _retry in range(5):
            try:
                vid, label, sources = self.items[i]
                train_mode = (self.split == 'train')
                if len(sources) == 1 and is_video(sources[0]):
                    vpath = sources[0]
                    n = decode_video_total_frames(vpath)
                    if n <= 0:
                        raise RuntimeError(f"Cannot decode video: {vpath}")
                    idxs = self._choose_indices(n, train=train_mode)
                    frames = []
                    tfm = self.tf_train if train_mode else self.tf_eval
                    cap = cv2.VideoCapture(str(vpath))
                    if not cap.isOpened():
                        raise RuntimeError(f"Cannot open video: {vpath}")
                    for j in idxs:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            raise RuntimeError(f"Read fail @ {vpath} frame {j}")
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        frames.append(tfm(img))
                    cap.release()
                    clip = torch.stack(frames, dim=0)  # (T,3,H,W)
                else:
                    # folder-of-frames case
                    n = len(sources)
                    idxs = self._choose_indices(n, train=train_mode)
                    frames = [ self._load_frame(sources[j], train=train_mode) for j in idxs ]
                    clip = torch.stack(frames, dim=0)

                return clip, torch.tensor(label, dtype=torch.float32), vid

            except Exception as e:
                # Log once, then pick a new random index and retry
                if _retry == 0:
                    sys.stderr.write(f"[DATA WARN] {e}\n")
                i = random.randrange(0, len(self.items))
                continue

        # Final fallback: raise after several retries
        raise RuntimeError("Too many failed video reads; dataset may contain many broken files.")

# ------------------- Frequency Embedding -------------------

class FrequencyEmbedding(nn.Module):
    """Global FFT radial-band embedding per frame."""
    def __init__(self, bands=8, embed_dim=128):
        super().__init__()
        self.bands = bands
        self.band_embed = nn.Parameter(torch.randn(bands, embed_dim))
        self.proj = nn.Sequential(nn.Linear(bands, 128), nn.ReLU(), nn.Linear(128, bands))

    @staticmethod
    def _radial_bins(h, w, bands, device):
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        cy, cx = (h-1)/2.0, (w-1)/2.0
        rr = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmax = rr.max() + 1e-6
        bins = torch.clamp((rr / rmax * bands).long(), 0, bands-1)
        return bins

    def forward(self, x):  # x: (B,T,3,H,W)
        x = x.float()
        B,T,C,H,W = x.shape
        r,g,b = x[:,:,0], x[:,:,1], x[:,:,2]
        gray = 0.299*r + 0.587*g + 0.114*b
        gray2 = gray.reshape(B*T,H,W)
        Freq = torch.fft.fftshift(torch.fft.fft2(gray2), dim=(-2,-1))
        mag = torch.abs(Freq) + 1e-6
        bins = self._radial_bins(H, W, self.bands, device=mag.device)

        mags = []
        for bb in range(self.bands):
            mask = (bins==bb).float()
            val = (mag * mask).sum(dim=(-2,-1)) / (mask.sum()+1e-6)
            mags.append(val)
        band_energy = torch.stack(mags, dim=-1)  # (B*T,bands)
        band_energy = (band_energy - band_energy.mean(dim=-1, keepdim=True)) / (band_energy.std(dim=-1, keepdim=True)+1e-6)

        logits = self.proj(band_energy)
        w = torch.softmax(logits, dim=-1)  # (B*T,bands)
        emb = w @ self.band_embed         # (B*T,embed)
        return emb.reshape(B,T,-1)

# ------------------- Model: VIDS-Guard -------------------

class TemporalAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim//4), nn.Tanh(), nn.Linear(dim//4, 1))
    def forward(self, seq):  # (B,T,D)
        w = torch.softmax(self.attn(seq).squeeze(-1), dim=1)  # (B,T)
        return (seq * w.unsqueeze(-1)).sum(1)  # (B,D)






class VIDSGuard(nn.Module):
    def __init__(self, img_size=224, freq_dim=128, T=8, layers=2, heads=8, emb_dim=512):
        super().__init__()
        self.T = T
        self.encoder = VGBBackbone(out_dim=emb_dim)
        self.freq = FrequencyEmbedding(bands=8, embed_dim=freq_dim)
        model_dim = emb_dim + freq_dim
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, batch_first=True,
            dim_feedforward=model_dim * 4, dropout=0.1, activation="gelu"
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pool = TemporalAttentionPool(model_dim)
        self.head = nn.Linear(model_dim, 1)

    def forward(self, x):  # x: (B,T,3,H,W)
        x = x.float()
        B, T = x.shape[:2]
        if x.ndim == 5 and x.shape[2] != 3 and x.shape[-1] == 3:
            # input came as (B, T, H, W, 3) -> convert to (B, T, 3, H, W)
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        _, _, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W).contiguous()  # (B*T, 3, H, W)
        if x_flat.shape[1] != 3 and x_flat.shape[-1] == 3:
            x_flat = x_flat.permute(0, 3, 1, 2).contiguous()
        f = self.encoder(x_flat)              # (B*T, emb_dim)
        f = f.reshape(B, T, -1)               # (B, T, emb_dim)
        g = self.freq(x)                      # (B, T, freq_dim)
        seq = torch.cat([f, g], dim=-1)       # (B, T, model_dim)
        seq = self.temporal(seq)              # (B, T, model_dim)
        z = self.pool(seq)                    # (B, model_dim)
        return self.head(z).squeeze(1)



# ------------------- Metrics & plots -------------------

def best_threshold(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    scores_sorted = y_score[order]
    P = y_true.sum(); N = len(y_true)-P
    tp = 0; fp = 0
    best_j, best_thr = -1.0, 0.5
    for i in range(len(scores_sorted)):
        if y_sorted[i] == 1: tp += 1
        else: fp += 1
        tpr = tp / (P + 1e-6)
        fpr = fp / (N + 1e-6)
        j = tpr - fpr
        thr = scores_sorted[i]
        if j > best_j:
            best_j, best_thr = j, thr
    return float(best_thr), float(best_j)

def confusion_at_threshold(y_true, y_score, thr):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= thr).astype(int)
    tn = int(((y_pred==0)&(y_true==0)).sum())
    tp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    acc = (tp+tn)/max(1,(tp+tn+fp+fn))
    return tn, fp, fn, tp, acc

def plot_curves(out_dir: Path, train_losses, val_aucs, val_accs):
    if not HAS_MPL: return
    # loss
    plt.figure(); plt.plot(range(len(train_losses)), train_losses)
    plt.title("Train Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.tight_layout(); plt.savefig(out_dir/"train_loss.png"); plt.close()
    # AUC
    plt.figure(); plt.plot(range(len(val_aucs)), val_aucs)
    plt.title("Val AUC"); plt.xlabel("Epoch"); plt.ylabel("AUC")
    plt.tight_layout(); plt.savefig(out_dir/"val_auc.png"); plt.close()
    # Acc
    plt.figure(); plt.plot(range(len(val_accs)), val_accs)
    plt.title("Val Accuracy @ 0.5"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.tight_layout(); plt.savefig(out_dir/"val_acc.png"); plt.close()

def plot_confusion(out_dir: Path, tn, fp, fn, tp, title="Confusion Matrix"):
    if not HAS_MPL: return
    M = np.array([[tn, fp],[fn, tp]], dtype=int)
    fig, ax = plt.subplots()
    im = ax.imshow(M, cmap='Blues')
    for (i, j), z in np.ndenumerate(M):
        ax.text(j, i, f"{z}", ha='center', va='center')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred 0','Pred 1']); ax.set_yticklabels(['True 0','True 1'])
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_dir/"confusion_best_thr.png"); plt.close()

# ------------------- Train / Eval -------------------

@torch.no_grad()
def evaluate(model, loader, device='cuda'):
    model.eval()
    video_scores: Dict[str, List[float]] = {}
    video_labels: Dict[str, int] = {}

    for clips, ys, vids in loader:
        clips = clips.to(device, dtype=torch.float32, non_blocking=True)
        logits = model(clips)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        ys_np = ys.numpy().astype(int)
        for v, p, y in zip(vids, probs, ys_np):
            video_scores.setdefault(v, []).append(float(p))
            video_labels[v] = int(y)

    # aggregate to video-level
    mean_scores = {v: float(np.mean(ps)) for v, ps in video_scores.items()}
    vids = list(mean_scores.keys())
    y_true = np.array([video_labels[v] for v in vids], dtype=int)
    y_score = np.array([mean_scores[v] for v in vids], dtype=float)

    # AUC
    auc = float('nan')
    if HAS_SK and len(np.unique(y_true)) == 2:
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = float('nan')

    # best threshold via Youden's J
    thr, j = best_threshold(y_true, y_score)

    # confusion at 0.5 and at best thr
    tn, fp, fn, tp, acc05 = confusion_at_threshold(y_true, y_score, 0.5)
    tn_b, fp_b, fn_b, tp_b, acc_b = confusion_at_threshold(y_true, y_score, thr)

    metrics = {
        'auc': auc if auc == auc else None,
        'acc@0.5': float(acc05),
        'best_thr': float(thr),
        'youdenJ': float(j),
        'acc@best_thr': float(acc_b),
        'tn@best_thr': int(tn_b), 'fp@best_thr': int(fp_b),
        'fn@best_thr': int(fn_b), 'tp@best_thr': int(tp_b)
    }
    # return raw arrays too so we can plot ROC/CM
    return metrics, mean_scores, (y_true, y_score)




def save_ckpt(path: Path, model, optim, scaler, epoch, best_auc, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optim': optim.state_dict() if optim else None,
        'scaler': scaler.state_dict() if scaler else None,
        'best_auc': best_auc,
        'args': vars(args)
    }, path)

def load_ckpt(path: Path, model, optim=None, scaler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    if optim and ckpt.get('optim'):
        optim.load_state_dict(ckpt['optim'])
    if scaler and ckpt.get('scaler'):
        scaler.load_state_dict(ckpt['scaler'])
    return ckpt

def plot_curves(out_dir: Path, train_losses, val_aucs, val_accs):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses)
    plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "loss_curve.png"); plt.close()

    # AUC
    plt.figure()
    plt.plot(range(len(val_aucs)), val_aucs)
    plt.xlabel("Epoch"); plt.ylabel("Val AUC"); plt.title("Validation AUC")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "val_auc_curve.png"); plt.close()

    # Acc@0.5
    plt.figure()
    plt.plot(range(len(val_accs)), val_accs)
    plt.xlabel("Epoch"); plt.ylabel("Val Acc@0.5"); plt.title("Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "val_acc_curve.png"); plt.close()


def plot_confusion(out_dir: Path, y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["real", "fake"])
    disp.plot(values_format='d', cmap='Blues', colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png"); plt.close()


def plot_roc(out_dir: Path, y_true, y_score, title="ROC Curve"):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png"); plt.close()

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    set_seed(args.seed)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    faces_root = Path(args.faces_root)
    items = discover_dataset(faces_root, broken_log=out_dir / "broken_videos.txt")


    if len(items) == 0:
        print(f"[ERROR] No data found under {faces_root}/{{real,fake}}", file=sys.stderr)
        sys.exit(1)

    train_items, val_items, _ = split_items(items, args.train_ratio, args.val_ratio, args.split_seed)

    train_set = ClipDataset(train_items, split='train', T=args.T, img_size=args.img_size,
                            clips_per_video=args.clips_per_video,
                            frame_drop_p=args.frame_drop_p, temporal_jitter=args.temporal_jitter,
                            jpeg_q=(args.jpeg_min_q, args.jpeg_max_q),
                            blur_p=args.blur_p, bc_strength=args.bc_strength)
#    val_set   = ClipDataset(val_items,   split='val',   T=args.T, img_size=args.img_size)
    val_set = ClipDataset(
    val_items, split='val', T=args.T, img_size=args.img_size,
    clips_per_video=args.eval_clips,
    frame_drop_p=0.0 if args.deterministic_val else args.frame_drop_p,
    temporal_jitter=0 if args.deterministic_val else args.temporal_jitter,
    jpeg_q=(args.jpeg_max_q, args.jpeg_max_q) if args.deterministic_val else (args.jpeg_min_q, args.jpeg_max_q),
    blur_p=0.0 if args.deterministic_val else args.blur_p, 
    bc_strength=0.0 if args.deterministic_val else args.bc_strength)




    print(f"Train clips/epoch: {len(train_set)}  Val videos: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False, persistent_workers=True)

    model = VIDSGuard(img_size=args.img_size, freq_dim=args.freq_dim,
                      T=args.T, layers=args.layers, heads=args.heads, emb_dim=512).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_epochs = args.epochs
    warmup = max(5, int(0.1 * total_epochs))
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    bce = nn.BCEWithLogitsLoss()
   

    # Cosine LR with warmup (simple & robust)
    import math
    total_epochs = args.epochs
    warmup = max(5, int(0.1 * total_epochs))
    def lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(max(1, warmup))
        progress = float(ep - warmup) / float(max(1, total_epochs - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    start_epoch, best_auc = 0, -1.0
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "metrics.csv"

    if args.resume and Path(args.resume).exists():
        ckpt = load_ckpt(Path(args.resume), model, optim, scaler)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_auc = ckpt.get('best_auc', -1.0)
        print(f"[INFO] Resumed from {args.resume}: epoch {start_epoch}, best AUC {best_auc:.4f}")

    # CSV header (create if missing)
    if not log_csv.exists():
        with open(log_csv, "w") as f:
            f.write("epoch,train_loss,val_auc,val_acc0.5,lr\n")

    train_losses, val_aucs, val_accs = [], [], []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        t0 = time.time()

        for clips, ys, _ in train_loader:
            clips = clips.to(device, dtype=torch.float32, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            try:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = model(clips)
                    loss = bce(logits, ys)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "CUDACachingAllocator" in str(e):
                    torch.cuda.empty_cache()
                    print("[WARN] Skipping batch due to GPU memory pressure.")
                    continue
                else:
                    raise

            optim.zero_grad(set_to_none=True)
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optim); scaler.update()
            else:
                loss.backward(); optim.step()
            losses.append(loss.item())

        # ---- EVAL ----
        val_metrics, val_scores, (y_true, y_score) = evaluate(model, val_loader, device=device)
        epoch_time = time.time() - t0

        # Log arrays for plotting
        train_losses.append(float(np.mean(losses)))
        val_aucs.append(float(val_metrics.get('auc', float('nan')) or float('nan')))
        val_accs.append(float(val_metrics.get('acc@0.5', float('nan'))))

        # One-line epoch print
        print(f"Epoch {epoch:03d}  loss {np.mean(losses):.4f}  "
              f"val_auc {val_metrics.get('auc', float('nan')):.4f}  "
              f"val_acc@0.5 {val_metrics.get('acc@0.5', float('nan')):.4f}  "
              f"time {epoch_time:.1f}s")
        sched.step()




        # CSV append
        with open(log_csv, "a") as f:
            f.write(f"{epoch},{np.mean(losses):.6f},{val_metrics.get('auc', float('nan'))},"
                    f"{val_metrics.get('acc@0.5', float('nan'))},{optim.param_groups[0]['lr']}\n")

        # Save per-epoch artifacts (metrics & curves)
        with open(out_dir / "val_scores.json", "w") as f: json.dump(val_scores, f, indent=2)
        with open(out_dir / "val_metrics.json", "w") as f: json.dump(val_metrics, f, indent=2)
        plot_curves(out_dir, train_losses, val_aucs, val_accs)

        # Confusion/ROC with best threshold
        if HAS_SK and len(np.unique(y_true)) == 2:
            thr = val_metrics['best_thr']
            y_pred = (y_score >= thr).astype(int)
            plot_confusion(out_dir, y_true, y_pred, title=f"Confusion (epoch {epoch}, thr={thr:.3f})")
            plot_roc(out_dir, y_true, y_score, title=f"ROC (epoch {epoch})")

        # ---- Checkpoints ----
        save_ckpt(out_dir / f"epoch_{epoch:03d}.pt", model, optim, scaler, epoch, best_auc, args)
        cur_auc = val_metrics.get('auc', -1.0) or -1.0
        if cur_auc > best_auc:
            best_auc = cur_auc
            save_ckpt(out_dir / "best.pt", model, optim, scaler, epoch, best_auc, args)

        # Step scheduler at epoch end
        sched.step()

    print("[DONE] Best val AUC:", best_auc)


# ------------------- CLI -------------------

def parse_args():
    p = argparse.ArgumentParser("VIDS-Guard training")
    p.add_argument("--faces_root", type=str, required=True,
                   help="Root with {real,fake}/*.mp4 (videos) or {real,fake}/<video-id>/*.jpg (frames).")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--resume", type=str, default="")

    # Model (custom backbone, no --enc)
    p.add_argument("--img_size", type=int, default=224, help="Input frame size.")
    p.add_argument("--freq_dim", type=int, default=128, help="Frequency dimension for temporal encoder.")
    p.add_argument("--layers", type=int, default=2, help="Number of temporal transformer layers.")
    p.add_argument("--heads", type=int, default=8, help="Number of attention heads in transformer.")
    p.add_argument("--T", type=int, default=8, help="Number of frames per video clip.")

    p.add_argument("--eval_clips", type=int, default=4, help="Clips per video at validation/test (deterministic).")

    p.add_argument("--deterministic_val", action="store_true", help="Disable all random jitter/drop in val.")





    # Train
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=5e-2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=41)

    # Split
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=41)

    # Temporal / Augs
    p.add_argument("--clips_per_video", type=int, default=4)
    p.add_argument("--frame_drop_p", type=float, default=0.1)
    p.add_argument("--temporal_jitter", type=int, default=1)
    p.add_argument("--jpeg_min_q", type=int, default=30)
    p.add_argument("--jpeg_max_q", type=int, default=90)
    p.add_argument("--blur_p", type=float, default=0.2)
    p.add_argument("--bc_strength", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
