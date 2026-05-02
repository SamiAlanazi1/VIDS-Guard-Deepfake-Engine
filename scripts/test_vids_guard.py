# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Test VIDS-Guard on an unseen dataset with real/ and fake/ folders.

- Loads architecture & args from a training checkpoint (epoch_XXX.pt).
- Builds a deterministic evaluation loader (multiple clips/video).
- Prints AUC, confusion/accuracy at a given --thr, and the per-video
  guess (real/fake) for EVERY video at that threshold.
- Also computes best threshold via Youden's J and prints per-video guesses.
- Saves confusion matrices as PNGs next to the checkpoint.

Usage (one line):
python3 -u //path to /test_vids_guard2.py \
  --faces_root /path to /UDS2MTCNN \
  --ckpt /path to //DFVDS/runs/vids_guard_vgb/epoch_011.pt \
  --eval_clips 4 --thr 0.55 --workers 8 --batch 8
"""

import os, sys, argparse, warnings, contextlib
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# plotting / metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# -------- Logging noise control (FFmpeg/OpenCV) --------
try:
    from cv2 import utils as cv2_utils
    cv2_utils.logging.setLogLevel(cv2_utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            yield

# --------- Basic IO / dataset helpers (videos or frame-folders) ----------
IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp'}
VID_EXTS = {'.mp4','.avi','.mov','.mkv','.webm'}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def is_video(p: Path) -> bool:
    return p.suffix.lower() in VID_EXTS

def probe_video_ok(video_path: Path, min_frames: int = 4) -> bool:
    try:
        if not video_path.exists() or video_path.stat().st_size < 1024:
            return False
        with suppress_stderr():
            cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        with suppress_stderr():
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return False
        return n >= min_frames
    except Exception:
        return False

def decode_video_total_frames(video_path: Path) -> int:
    with suppress_stderr():
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

class EvalTransform:
    def __init__(self, img_size=224):
        self.img_size = img_size
    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img).astype(np.float32)/255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        arr = (arr - mean)/std
        arr = arr.transpose(2,0,1).astype(np.float32)
        return torch.from_numpy(arr)

def discover_dataset(root: Path) -> List[Tuple[str,int,List[Path]]]:
    """
    Accepts either:
      A) root/{real,fake}/<video-id>/*.{jpg,png,...}   (folders of frames)
      B) root/{real,fake}/*.{mp4,avi,mov,mkv,webm}     (video files)
    Returns: (video_id, label_int, sources)
      - For frame folders: sources = list of frame Paths
      - For videos:        sources = [single Path to video file]
    """
    items: List[Tuple[str,int,List[Path]]] = []
    for label_name, label_int in [('real',0), ('fake',1)]:
        label_dir = root / label_name
        if not label_dir.exists():
            warnings.warn(f"Missing folder: {label_dir}")
            continue

        # video files
        vids = [p for p in sorted(label_dir.iterdir()) if p.is_file() and is_video(p)]
        for vp in vids:
            if probe_video_ok(vp):
                vid_id = f"{label_name}/{vp.stem}"
                items.append((vid_id, label_int, [vp]))
        # folders of frames
        for vd in sorted([d for d in label_dir.iterdir() if d.is_dir()]):
            frames = sorted([p for p in vd.iterdir() if p.is_file() and is_image(p)])
            if frames:
                items.append((f"{label_name}/{vd.name}", label_int, frames))
    return items

class ClipDataset(Dataset):
    """
    Deterministic evaluation dataset.
    For videos: uniformly sample T frames per clip; repeat 'clips_per_video' times with fixed offsets.
    For frame-folders: same logic on image lists.
    """
    def __init__(self, items, T=8, img_size=224, clips_per_video=4):
        self.items = items
        self.T = T
        self.img_size = img_size
        self.clips_per_video = clips_per_video
        self.tf = EvalTransform(img_size)

        # build an index mapping of (video_idx, which_clip)
        self.index = []
        for i in range(len(items)):
            for k in range(clips_per_video):
                self.index.append((i, k))

    def __len__(self):
        return len(self.index)

    def _sample_indices(self, n:int, k:int) -> List[int]:
        """Uniform deterministic sampling with per-clip phase shift."""
        if n <= 0:
            return [0]*self.T
        step = max(1, n // self.T)
        start = (k * step) % max(1, n - self.T*step + 1)
        idxs = [min(n-1, start + t*step) for t in range(self.T)]
        return idxs

    def __getitem__(self, idx):
        i, k = self.index[idx]
        vid, label, sources = self.items[i]

        if len(sources) == 1 and is_video(sources[0]):
            vpath = sources[0]
            n = decode_video_total_frames(vpath)
            if n <= 0:
                raise RuntimeError(f"Cannot decode video: {vpath}")
            idxs = self._sample_indices(n, k)
            with suppress_stderr():
                cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {vpath}")
            frames=[]
            for j in idxs:
                with suppress_stderr():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
                    ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Read fail @ {vpath} frame {j}")
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(self.tf(img))
            cap.release()
            clip = torch.stack(frames, dim=0)
        else:
            frames = [p for p in sources]
            n = len(frames)
            idxs = self._sample_indices(n, k)
            imgs = []
            for j in idxs:
                img = Image.open(frames[j]).convert('RGB')
                imgs.append(self.tf(img))
            clip = torch.stack(imgs, dim=0)

        return clip, torch.tensor(label, dtype=torch.float32), vid

# ---------------- Load model from checkpoint ----------------
def load_ckpt_build_model(ckpt_path: Path, device='cuda'):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = ckpt.get('args', {})
    # Import model definition to ensure exact match
    sys.path.insert(0, str(Path(__file__).parent))
    from train_vids_guard import VIDSGuard
    model = VIDSGuard(
        img_size=int(args.get('img_size', 224)),
        freq_dim=int(args.get('freq_dim', 128)),
        T=int(args.get('T', 8)),
        layers=int(args.get('layers', 2)),
        heads=int(args.get('heads', 8)),
        emb_dim=512
    )
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device)
    model.eval()
    return model, args

# ---------------- Evaluation utilities ----------------
@torch.no_grad()
def eval_on_dataset(model, loader, device='cuda'):
    """
    Returns:
      vids   : list[str] in deterministic order
      y_true : np.ndarray[int]  (0=real, 1=fake)
      y_score: np.ndarray[float] (mean probability per video)
      auc    : float (NaN if not computable)
    """
    model.eval()
    video_scores: Dict[str, List[float]] = {}
    video_labels: Dict[str, int] = {}
    for clips, ys, vids in loader:
        clips = clips.to(device, dtype=torch.float32, non_blocking=True)
        logits = model(clips)          # (B,)
        probs = torch.sigmoid(logits).cpu().numpy()
        ys_np = ys.numpy().astype(int)
        for v, p, y in zip(vids, probs, ys_np):
            video_scores.setdefault(v, []).append(float(p))  # insertion order preserved
            video_labels[v] = int(y)

    # Aggregate clip probs -> video prob, keep insertion order
    vids = list(video_scores.keys())
    y_true = np.array([video_labels[v] for v in vids], dtype=int)
    y_score = np.array([np.mean(video_scores[v]) for v in vids], dtype=float)

    auc = float('nan')
    if len(np.unique(y_true)) == 2:
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
    return vids, y_true, y_score, auc

def youden_best_thr(y_true, y_score):
    # Finds the threshold that maximizes Youden's J = TPR - FPR
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    scores = y_score[order]
    P = int(y_true.sum()); N = len(y_true) - P
    tp = fp = 0
    best_j = -1.0
    best_thr = 0.5
    for i in range(len(scores)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / (P + 1e-9)
        fpr = fp / (N + 1e-9)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thr = scores[i]
    return float(best_thr), float(best_j)

def confusion_and_acc(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(1, (tp+tn+fp+fn))
    return cm, acc, y_pred

def save_confusion_fig(cm, out_png, title):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["real","fake"])
    disp.plot(values_format='d', cmap='Blues', colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ---------------- CLI & main ----------------
def parse_args():
    p = argparse.ArgumentParser("VIDS-Guard testing on unseen dataset")
    p.add_argument("--faces_root", type=str, required=True,
                   help="Unseen dataset root with {real,fake}/ (videos or frame-folders).")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Checkpoint path (epoch_XXX.pt or best.pt).")
    p.add_argument("--eval_clips", type=int, default=4,
                   help="Clips per video at eval.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--thr", type=float, default=0.55,
                   help="Decision threshold for accuracy/CM.")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch of clips.")
    return p.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = Path(args.ckpt)
    out_dir = ckpt_path.parent  # save figures next to the checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build model from checkpoint
    model, train_args = load_ckpt_build_model(ckpt_path, device=device)
    T = int(train_args.get('T', 8))
    img_size = int(train_args.get('img_size', 224))

    # Discover unseen dataset & loader
    items = discover_dataset(Path(args.faces_root))
    if len(items) == 0:
        print(f"[ERROR] No data under {args.faces_root}/{{real,fake}}", file=sys.stderr)
        sys.exit(1)

    ds = ClipDataset(items, T=T, img_size=img_size, clips_per_video=args.eval_clips)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        drop_last=False, persistent_workers=(args.workers>0))

    print(f"Device: {device}")
    print(f"Unseen videos: {len(ds.items)}  (clips: {len(ds)})")

    # Evaluate
    vids, y_true, y_score, auc = eval_on_dataset(model, loader, device=device)

    # Accuracy @ given thr
    cm_thr, acc_thr, y_pred_thr = confusion_and_acc(y_true, y_score, args.thr)
    save_confusion_fig(cm_thr, out_dir / "confusion_unseen_thr.png",
                       f"Confusion (thr={args.thr:.3f})")

    # ---------- Print per-video results ----------
    def lbl(i): return "real" if i == 0 else "fake"

    tn, fp, fn, tp = cm_thr.ravel()
    print(f"\n=== Results @ thr={args.thr:.2f} ===")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"Accuracy@{args.thr:.2f}: {acc_thr*100:.2f}%")

    print(f"\n--- Per-video predictions (thr={args.thr:.2f}) ---")
    for v, y_t, y_p, s in zip(vids, y_true, y_pred_thr, y_score):
        print(f"video: {v}\ttrue={lbl(y_t)}\tguess={lbl(y_p)}\tscore={s:.4f}")

    # Best thr via Youden's J
    best_thr, _ = youden_best_thr(y_true, y_score)
    cm_best, acc_best, y_pred_best = confusion_and_acc(y_true, y_score, best_thr)
    save_confusion_fig(cm_best, out_dir / "confusion_unseen_best.png",
                       f"Confusion (best thr={best_thr:.3f})")

    btn, bfp, bfn, btp = cm_best.ravel()
    print(f"\n=== Results @ best thr={best_thr:.3f} (Youden) ===")
    print(f"Confusion: TN={btn}  FP={bfp}  FN={bfn}  TP={btp}")
    print(f"Accuracy@best: {acc_best*100:.2f}%")

    print(f"\n--- Per-video predictions (best thr={best_thr:.3f}) ---")
    for v, y_t, y_p, s in zip(vids, y_true, y_pred_best, y_score):
        print(f"video: {v}\ttrue={lbl(y_t)}\tguess={lbl(y_p)}\tscore={s:.4f}")

    print("\n[SAVED] confusion_unseen_thr.png")
    print("[SAVED] confusion_unseen_best.png")

if __name__ == "__main__":
    main()

