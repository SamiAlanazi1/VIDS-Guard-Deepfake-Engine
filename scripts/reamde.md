# Scripts Overview

This directory contains all preprocessing, training, and evaluation scripts required to reproduce the experiments reported in the **VIDS-Guard** study.  
Scripts are designed to be executed sequentially, following the dataset preparation and model training workflow.

---

## Step 1: Face Extraction

### `video_face_extractor_mtcnn.py`

Extracts face-centric video clips from raw videos using the MTCNN face detector.

**Purpose**
- Detect faces in each frame
- Crop and align face regions
- Save face-only video clips for efficient training

**Input**
- Raw video files (real and fake)

**Output**
- Face-cropped videos organized under `{real, fake}` directories

**Command**
```bash
python video_face_extractor_mtcnn.py \
    --videos_location [PATH_TO_RAW_VIDEOS] \
    --face_videos_location [PATH_TO_FACE_VIDEOS]



## Step 2: Dataset Indexing

### `index_face_data_mtcnn.py`

Indexes the extracted face videos and generates the metadata required for dataset splitting and training.

**Purpose**
- Scan the face-video directory structure
- Assign labels based on `{real, fake}` subfolders
- Store video paths and identifiers for reproducible splitting

**Input**
- Face-cropped videos generated in Step 1 (organized under `{real, fake}`)

**Output**
- `videos.pkl`: serialized index containing video metadata

**Command**
```bash
python index_face_data_mtcnn.py



## Step 3: Dataset Preprocessing and Splitting

### `preprocess_mtcnn.py`

Generates deterministic training, validation, and test splits from the indexed dataset.

**Purpose**
- Enforce an 80/10/10 video-level split
- Maintain class balance across splits
- Support subject-disjoint partitioning where applicable

**Input**
- `videos.pkl` generated in Step 2

**Output**
- Updated `videos.pkl` with split annotations

**Command**
```bash
python preprocess_mtcnn.py


## Step 4: Model Training

### `train_vids_guard.py`

Trains the VIDS-Guard model using preprocessed face video clips.

**Purpose**
- Sample temporal clips from face videos
- Apply multi-stream forensic feature extraction (spatial, chroma, and frequency)
- Train the temporal transformer-based classification model

**Input**
- Face-cropped videos with split annotations (`videos.pkl`)

**Output**
- Trained model checkpoints
- Training logs and validation metrics

**Key Parameters**
- `--T`: temporal window length  
- `--clips_per_video`: number of clips sampled per video  
- `--batch`: batch size  
- `--epochs`: number of training epochs  
- `--amp`: enable mixed-precision training  

**Command**
```bash
python train_vids_guard.py \
    --faces_root [PATH_TO_FACE_VIDEOS] \
    --out [OUTPUT_PATH]/runs/vids_guard_vgb \
    --T 16 \
    --clips_per_video 4 \
    --batch 8 \
    --epochs 15 \
    --workers 2 \
    --amp


## Step 5: Model Evaluation

### `test_vids_guard.py`

Evaluates a trained VIDS-Guard model on internal or external test datasets.

**Purpose**
- Perform video-level inference on face-cropped videos
- Aggregate predictions across multiple temporal clips
- Compute and report evaluation metrics

**Input**
- Face-cropped videos from the test or unseen dataset
- Trained model checkpoint

**Output**
- Evaluation metrics (Accuracy, ROC-AUC)
- Confusion matrix
- ROC curve

**Command**
```bash
python test_vids_guard.py \
    --faces_root [PATH_TO_UNSEEN_FACE_VIDEOS] \
    --ckpt [OUTPUT_PATH]/runs/vids_guard_vgb/epoch_011.pt \
    --eval_clips 4 \
    --thr 0.55 \
    --workers 8

