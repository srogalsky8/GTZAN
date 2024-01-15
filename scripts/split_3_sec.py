import librosa
import soundfile as sf
import os

def split_audio_files(source_folder, dest_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=None)

                # divide into 10 equal chunks
                num_chunks = 10
                # length of each chunk in samples
                chunk_length = len(y) // num_chunks

                for i in range(num_chunks):
                    start = i * chunk_length
                    end = start + chunk_length
                    # if there's any remainder samples, add it to the end of the last chunk
                    if i == num_chunks - 1:
                        end = len(y)
                    chunk = y[start:end]

                    new_file_name = f"{file[:-4]}_chunk_{i}.wav"
                    save_path = os.path.join(dest_folder, os.path.relpath(root, source_folder), new_file_name)

                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    sf.write(save_path, chunk, sr)
                    
base_dir = 'data/split_wav'
split_folders = ['test', 'train', 'val']

for folder in split_folders:
    source_folder = os.path.join(base_dir, folder)
    dest_folder = os.path.join(base_dir, f"{folder}_3sec")
    split_audio_files(source_folder, dest_folder)
