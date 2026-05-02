from pathlib import Path
import glob, numpy as np, pandas as pd
from tqdm import tqdm

folders_len = 50

# Deterministic listing
real_paths = sorted(glob.glob('/path to /DFVDSMTCNN/real/*'))
fake_paths = sorted(glob.glob('//path to /DFVDSMTCNN/fake/*'))

# Build per-class DataFrames
df_real = pd.DataFrame({'path': [Path(p) for p in real_paths], 'label': False})
df_fake = pd.DataFrame({'path': [Path(p) for p in fake_paths], 'label': True})

# Concatenate (keeps all samples even if imbalanced)
df_videos = pd.concat([df_real, df_fake], ignore_index=True)

# Optional: shuffle for randomness (set seed for reproducibility)
df_videos = df_videos.sample(frac=1, random_state=42).reset_index(drop=True)


# Assign folders in a class-balanced round-robin way
df_videos['folder'] = -1
for label_val in [False, True]:
    idxs = df_videos.index[df_videos['label'] == label_val].tolist()
    for k, i in enumerate(idxs):
        df_videos.at[i, 'folder'] = k % folders_len

# ---- (Optional) collect metadata exactly as you had it) ----
import av
def extract_meta_av(path: str):
    try:
        video = av.open(path)
        vs = video.streams.video[0]
        return vs.height, vs.width, vs.frames
    except Exception as e:
        print(f'Error: {path}\n{e}')
        return 0, 0, 0

heights, widths, frames = [], [], []
for p in tqdm(df_videos['path'].astype(str), desc='Metadata'):
    h, w, f = extract_meta_av(p)
    heights.append(h); widths.append(w); frames.append(f)

df_videos['height'] = np.asarray(heights, dtype=np.uint16)
df_videos['width']  = np.asarray(widths,  dtype=np.uint16)
df_videos['frames'] = np.asarray(frames,  dtype=np.uint16)

# Save
out_path = '//path to /DFVDSMTCNN/videos.pkl'
df_videos.to_pickle(out_path)

print(f"Real videos: {(~df_videos['label']).sum()}")
print(f"Fake videos: {df_videos['label'].sum()}")
