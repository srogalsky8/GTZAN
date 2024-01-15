import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def create_spectrograms(source_folder, dest_folder):
    # for each directory
    for root, dirs, files in os.walk(source_folder):
        # for each file
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=None)
                
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

                # convert to log scale
                S_dB = librosa.power_to_db(S, ref=np.max)

                plt.figure(figsize=(3, 3))
                librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')

                # Remove axes labels, titles, and colorbars
                plt.axis('off')

                # save file
                new_file_name = f"{file[:-4]}.png"
                save_path = os.path.join(dest_folder, os.path.relpath(root, source_folder), new_file_name)

                # create directory in target split folder
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()

base_dir = 'data/split_wav'
split_folders = ['test_3sec', 'train_3sec', 'val_3sec']

for folder in split_folders:
    source_folder = os.path.join(base_dir, folder)
    dest_folder = os.path.join(base_dir, f"{folder}_spectrograms")
    create_spectrograms(source_folder, dest_folder)
