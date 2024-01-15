import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

def split_dataset(base_dir, train_size=0.8, val_size=0.1, test_size=0.1, random_seed=8):
    np.random.seed(random_seed)

    original_data_dir = os.path.join(base_dir, 'images_original')
    split_dir = os.path.join(base_dir, 'split')
    train_dir = os.path.join(split_dir, 'train')
    val_dir = os.path.join(split_dir, 'val')
    test_dir = os.path.join(split_dir, 'test')

    # create the new split directories
    for directory in [split_dir, train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    num_train_images = 0
    num_val_images = 0
    num_test_images = 0

    # get unique genres
    classes = [d.name for d in os.scandir(original_data_dir) if d.is_dir()]
    # for each genre, create a directory for each train/val/test
    for cls in classes:
        for directory in [train_dir, val_dir, test_dir]:
            class_dir = os.path.join(directory, cls)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

        class_dir = os.path.join(original_data_dir, cls)
        images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(images)

        # do train_val_test_split
        train_images, test_images = train_test_split(images, train_size=train_size, random_state=random_seed) # first split off train
        val_size_adjusted = val_size / (val_size + test_size)  # proportion of remaining that is val
        val_images, test_images = train_test_split(test_images, train_size=val_size_adjusted, random_state=random_seed) # split remainder into test/val

        def copy_images(images, source_dir, target_dir):
            for img in images:
                src_path = os.path.join(source_dir, img)
                dst_path = os.path.join(target_dir, img)
                shutil.copy(src_path, dst_path)

        copy_images(train_images, class_dir, os.path.join(train_dir, cls))
        num_train_images += len(train_images)
        copy_images(val_images, class_dir, os.path.join(val_dir, cls))
        num_val_images += len(val_images)
        copy_images(test_images, class_dir, os.path.join(test_dir, cls))
        num_test_images += len(test_images)

    print(f"Dataset split complete.")
    print(f"Total images in train set: {num_train_images}")
    print(f"Total images in validation set: {num_val_images}")
    print(f"Total images in test set: {num_test_images}")

# Usage
split_dataset(base_dir='data', train_size=0.8, val_size=0.1, test_size=0.1, random_seed=8)
