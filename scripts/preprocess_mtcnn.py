
import pandas as pd

# Load original videos.pkl
df_videos = pd.read_pickle('/path to videos.pkl')

# 80% train, 10% validation, 10% test
df_train = df_videos[df_videos.folder.isin(range(0, 40))]
df_val   = df_videos[df_videos.folder.isin(range(40, 45))]
df_test  = df_videos[df_videos.folder.isin(range(45, 50))]

# Save splits
df_train.to_pickle('/pathe to train.pkl')
df_val.to_pickle('/path to val.pkl')
df_test.to_pickle('/path to test.pkl')
