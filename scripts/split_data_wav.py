import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

def split_dataset(base_dir, train_size=0.8, val_size=0.1, test_size=0.1, random_seed=8):
    np.random.seed(random_seed)

    original_data_dir = os.path.join(base_dir, 'genres_original')
    split_wav_dir = os.path.join(base_dir, 'split_wav')
    train_dir = os.path.join(split_wav_dir, 'train')
    val_dir = os.path.join(split_wav_dir, 'val')
    test_dir = os.path.join(split_wav_dir, 'test')

    # create the new split directories
    for directory in [split_wav_dir, train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    num_train_files = 0
    num_val_files = 0
    num_test_files = 0

    # get unique genres
    genres = [d.name for d in os.scandir(original_data_dir) if d.is_dir()]
    # for each genre, create a directory for each train/val/test
    for genre in genres:
        for directory in [train_dir, val_dir, test_dir]:
            genre_dir = os.path.join(directory, genre)
            if not os.path.exists(genre_dir):
                os.makedirs(genre_dir)

        genre_dir = os.path.join(original_data_dir, genre)
        files = [file for file in os.listdir(genre_dir) if file.lower().endswith('.wav')]
        print(files)

        # do train_val_test_split
        train_files, test_files = train_test_split(files, train_size=train_size, random_state=random_seed) # first split off train
        val_size_adjusted = val_size / (val_size + test_size)  # proportion of remaining that is val
        val_files, test_files = train_test_split(test_files, train_size=val_size_adjusted, random_state=random_seed) # split remainder into test/val

        def copy_files(files, source_dir, target_dir):
            for file in files:
                src_path = os.path.join(source_dir, file)
                dst_path = os.path.join(target_dir, file)
                shutil.copy(src_path, dst_path)

        copy_files(train_files, genre_dir, os.path.join(train_dir, genre))
        num_train_files += len(train_files)
        copy_files(val_files, genre_dir, os.path.join(val_dir, genre))
        num_val_files += len(val_files)
        copy_files(test_files, genre_dir, os.path.join(test_dir, genre))
        num_test_files += len(test_files)

    # Print out the count of files in each set
    print(f"Dataset split complete.")
    print(f"Total .wav files in train set: {num_train_files}")
    print(f"Total .wav files in validation set: {num_val_files}")
    print(f"Total .wav files in test set: {num_test_files}")

# Usage
split_dataset(base_dir='data', train_size=0.8, val_size=0.1, test_size=0.1, random_seed=8)
