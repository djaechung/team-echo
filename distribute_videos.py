import pandas as pd
import os
import shutil

FileListCSV = pd.read_csv("FileList.csv")

train_vids = []
val_vids = []
test_vids = []

train_vid_names = set(FileListCSV[FileListCSV['Split'] == 'TRAIN']['FileName'].tolist())
val_vid_names = set(FileListCSV[FileListCSV['Split'] == 'VAL']['FileName'].tolist())
test_vid_names = set(FileListCSV[FileListCSV['Split'] == 'TEST']['FileName'].tolist())

print(len(train_vid_names))
print(len(val_vid_names))
print(len(test_vid_names))

def transfer_file(names, phase='train'):
    if phase not in os.listdir('datasets'):
        os.mkdir(f"datasets/{phase}")
    for i, vid in enumerate(names):
        if i % 50 == 0:
            print("i is: ", i)
        vid_file = vid + '.avi'
        destination_folder = f"datasets/{phase}"
        shutil.copy(f"Videos/{vid_file}", destination_folder)
    print("done")

transfer_file(train_vid_names, phase='train')
transfer_file(val_vid_names, phase='val')
transfer_file(test_vid_names, phase='test')