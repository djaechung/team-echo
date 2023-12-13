import cv2
from torch.utils.data import Dataset
import numpy as np
import torch
import random
import os
import pandas as pd
import subprocess

FileListCSV = pd.read_csv("FileList.csv")

def avi_to_tensor(video_file, max_frames=None):
    # Initialize a list to store video frames
    frames = []

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Read frames from the video file
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Transpose the frame shape from (height, width, channel) to (channel, height, width)
        frame_t = np.transpose(frame_rgb, (2, 0, 1))

        frames.append(frame_t)

        # Stop reading frames if maximum number of frames is reached
        if max_frames is not None and len(frames) >= max_frames:
            break

    # Release the video file
    cap.release()

    # Stack the frames to create a 4D numpy array
    video_array = np.stack(frames, axis=0)

    # Convert the numpy array to a PyTorch tensor
    video_tensor = torch.from_numpy(video_array).float()

    return video_tensor
  
class VideoDataset(Dataset):
    def __init__(self, video_tensors, labels, video_ids, ef_vals):
        self.video_tensors = video_tensors
        self.labels = labels
        self.video_ids = video_ids
        self.ef_vals = ef_vals

    def __len__(self):
        return len(self.video_tensors)

    def __getitem__(self, index):
        video_data = self.video_tensors[index]
        label = self.labels[index]
        id = self.video_ids[index]
        ef_val = self.ef_vals[index]
        return video_data, label, id, ef_val

def batch_data(data_dir, batch_size, video_names, videos_used_so_far, phase = 'train', rebalancing=False):
  location_of_videos = f"{data_dir}/{phase}"
  if rebalancing == False:
      destinated_batched_tensor_folder = f"{data_dir}/{phase}_batched_downsampled"
  else:
      destinated_batched_tensor_folder = f"{data_dir}/{phase}_batched_downsampled_rebalanced"

  random.shuffle(video_names)

  num_videos = len(video_names)
  num_batches = num_videos // batch_size
  batches = []

  video_names = [video for video in video_names if video not in videos_used_so_far]

  print("we are creating batches")
  for i in range(num_batches):
      start = i * batch_size
      end = min((i + 1) * batch_size, num_videos)
      batches.append(video_names[start:end])
  
  if num_videos % batch_size != 0:
    remainder_batch = video_names[num_batches * batch_size:]
    batches.append(remainder_batch)
  
  print("done creating batches")

  print(os.listdir())

  print(os.listdir("datasets"))
  print("os.listdir(destinated_batched_tensor_folder) is: ", os.listdir(destinated_batched_tensor_folder))


  if len(os.listdir(destinated_batched_tensor_folder)) > 0:
      for file in os.listdir(destinated_batched_tensor_folder):
          os.remove(f"{destinated_batched_tensor_folder}/{file}")

  for i, batch in enumerate(batches):
    if i % 10 == 0:
      print(f"we are on the {i}th batch")
    batch_name = f'{phase}_batch_{i}.pt'
    if batch_name in os.listdir(destinated_batched_tensor_folder):
      print("batch_name already in the folder")
      continue
    batch_X_list = []
    batch_y_list = []
    video_ids = []
    running_efs = []

    for video in batch:
      videos_used_so_far.append(video)
      tensor = avi_to_tensor(location_of_videos + '/' + video + '.avi')
      tensor = tensor[::4]
      batch_X_list.append(tensor)
      batch_y_list.append(FileListCSV[FileListCSV['FileName'] == video]['EFBelow40'].item())
      video_ids.append(video)
      running_efs.append(FileListCSV[FileListCSV['FileName'] == video]['EF'].item())
    
    mini_dataset = VideoDataset(batch_X_list, batch_y_list, video_ids, running_efs)
    torch.save(mini_dataset, destinated_batched_tensor_folder + '/' + batch_name)

  return batches