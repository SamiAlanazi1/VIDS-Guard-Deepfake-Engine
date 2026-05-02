#mtcnn
import cv2
import tqdm
import os
from pathlib import Path
import argparse
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


parser = argparse.ArgumentParser()
parser.add_argument('--videos_location', type=str, help='main path to input videos')
parser.add_argument('--face_videos_location', type=str, help='main path to output videos')
args = parser.parse_args()

videos_location = args.videos_location
face_videos_location = args.face_videos_location


def extract_highest_prob_face(input_video_path, output_video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    input_video = cv2.VideoCapture(input_video_path)

    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_video.get(cv2.CAP_PROP_FPS)

    output_video = cv2.VideoWriter(output_video_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (frame_width, frame_height))

    while True:
        ret, frame = input_video.read()

        if not ret:
            break
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(rgb_image)
        if boxes is not None :
            boxes, probs =   boxes.tolist(), probs.tolist()

            max_acc = max(probs)
            max_acc_index = probs.index(max_acc)

            largest_face = boxes[max_acc_index]
            x1, y1, x2, y2 = largest_face
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face_frame = frame[y1:y2, x1:x2]
            try:
                face_frame_resized = cv2.resize(face_frame, (frame_width, frame_height))
                output_video.write(face_frame_resized)
            except:
                output_video.write(frame)

        else:
            output_video.write(frame)

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

if not os.path.exists(face_videos_location):
  os.makedirs(face_videos_location)

for vid_type in ['real', 'fake']:
  videos_location_sub = Path(videos_location) / vid_type
  face_videos_location_sub = Path(face_videos_location ) / vid_type
  if not os.path.exists(face_videos_location_sub):
    os.makedirs(face_videos_location_sub)

  for vid_name in tqdm.tqdm(os.listdir(videos_location_sub)):
    input_video_path = videos_location_sub / vid_name
    output_video_path = face_videos_location_sub / vid_name
    if output_video_path.exists() and output_video_path.stat().st_size > 0:
        continue
    try:
        extract_highest_prob_face(input_video_path, output_video_path)

    except Exception as e:
        print(f"[Warning] Skipping {vid_name} due to error: {e}")

