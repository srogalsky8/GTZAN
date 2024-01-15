import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from ImageFolderWithPaths import ImageFolderWithPaths
from collections import defaultdict
from scipy.stats import mode
import os
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


resize_scale = 0.5

def custom_transform(image):
    # crop image
    image = image.crop((55, 35, 390, 253))
    return transforms.ToTensor()(image)

def compute_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0
    for images, label, path in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images.size(0)
    mean /= total_images_count
    std /= total_images_count
    return mean, std

# load from data/images_original - do the splitting
def load_data(resize_images=False, val_size=None, random_seed=8, batch_size=10):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # convert to grayscale and crop
    transforms_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(custom_transform),  # crop transform
    ]
    if resize_images:
        transforms_list.append(transforms.Resize((int(218*resize_scale), int(335*resize_scale))))
    pre_transform = transforms.Compose(transforms_list)

    data_dir = 'data/images_original'
    dataset = datasets.ImageFolder(root=data_dir, transform=pre_transform)

    # calc size of each set
    total_size = len(dataset)
    if val_size is not None:
        val_size = math.ceil(val_size * total_size)
    else:
        val_size = 0
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size - val_size

    # split data
    train_dataset, val_test_dataset = random_split(dataset, [train_size, total_size - train_size])
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    # compute mean and std
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    mean, std = compute_mean_std(train_loader)

    # normalize transform
    transforms_list.append(transforms.Normalize(mean, std))
    data_transform = transforms.Compose(transforms_list)
    train_dataset.dataset.transform = data_transform
    val_dataset.dataset.transform = data_transform
    test_dataset.dataset.transform = data_transform

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_size > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def load_data_from_split(resize_images=False, random_seed=8, batch_size=20, split_songs=False):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    transforms_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(custom_transform),
    ]
    if resize_images:
        transforms_list.append(transforms.Resize((int(218*resize_scale), int(335*resize_scale))))

    # No normalization at this point, to be computed later
    data_transform = transforms.Compose(transforms_list)

    if (split_songs):
        # use 3 second clips dataset
        train_data_dir = 'data/split_wav/train_3sec_spectrograms'
        val_data_dir = 'data/split_wav/val_3sec_spectrograms'
        test_data_dir = 'data/split_wav/test_3sec_spectrograms'
    else:
        # use full song dataset
        train_data_dir = 'data/split/train'
        val_data_dir = 'data/split/val'
        test_data_dir = 'data/split/test'

    train_dataset = ImageFolderWithPaths(root=train_data_dir, transform=data_transform)
    val_dataset = ImageFolderWithPaths(root=val_data_dir, transform=data_transform)
    test_dataset = ImageFolderWithPaths(root=test_data_dir, transform=data_transform)

    # compute mean and std
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    mean, std = compute_mean_std(train_loader)

    # normalization transform
    normalization_transform = transforms.Normalize(mean, std)
    train_dataset.transform = transforms.Compose([data_transform, normalization_transform])
    val_dataset.transform = transforms.Compose([data_transform, normalization_transform])
    test_dataset.transform = transforms.Compose([data_transform, normalization_transform])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


# Convert pytorch to numpy (for classical models)
def dataset_to_numpy(loader):
    images = []
    labels = []
    for img, label in loader:
        images.append(img.view(img.size(0), -1).numpy())
        labels.append(label.numpy())
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images, labels

def load_numpy_data(resize_images=False, val_size=None):
    train_loader, test_loader, val_loader = load_data(resize_images, val_size=val_size)
    X_train, y_train = dataset_to_numpy(train_loader)
    X_test, y_test = dataset_to_numpy(test_loader)
    X_val, y_val = dataset_to_numpy(val_loader)
    return X_train, y_train, X_test, y_test, X_val, y_val

def load_numpy_data_from_split(resize_images=False):
    train_loader, test_loader, val_loader = load_data_from_split(resize_images)
    X_train, y_train = dataset_to_numpy(train_loader)
    X_val, y_val = dataset_to_numpy(val_loader)
    X_test, y_test = dataset_to_numpy(test_loader)
    return X_train, y_train, X_val, y_val, X_test, y_test


def test_model(net, data_loader, device, criterion):
    net.eval()

    accuracy = None
    loss = None
    running_loss = 0
    avg_loss = None

    with torch.no_grad():
        # eval and calculate accuracy
        correct = 0
        total = 0
        for data in data_loader:
            images, labels, paths = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(data_loader)
    
    return accuracy, avg_loss

def parse_original_identifier(path):
    # Get the original file from the chunked filename
    # ex: 'data/split_wav/test_3sec_spectrograms/metal/metal.00024_chunk_6.png' -> 'metal.00024'
    basename = os.path.basename(path)
    return '_'.join(basename.split('_')[:-1])

from collections import defaultdict
from scipy.stats import mode

def test_model_aggregate(net, data_loader, device):
    net.eval()
    predictions = defaultdict(list)
    labels_aggregated = defaultdict(list)

    with torch.no_grad():
        for images, labels, paths in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            for path, pred, label in zip(paths, predicted, labels):
                identifier = parse_original_identifier(path)
                predictions[identifier].append(pred.cpu().item())
                labels_aggregated[identifier].append(label.cpu().item())

    # get mode of predictions for each original file
    correct = 0
    total = 0
    for identifier in predictions:
        mode_prediction = mode(predictions[identifier]).mode[0]
        mode_label = mode(labels_aggregated[identifier]).mode[0]
        total += 1
        if mode_prediction == mode_label:
            correct += 1

    accuracy = 100 * correct / total
    return accuracy


def plot_loss(training_losses, validation_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(training_losses)+1), training_losses, label='Training Loss')
    plt.plot(range(1, len(validation_losses)+1), validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.show()


def train_loop(net, device, train_loader, val_loader, model_path = 'model.pth', criterion=nn.CrossEntropyLoss(), split_songs = False, num_epochs=20, optimizer=None, scheduler=None):
    random_seed = 8
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    net.to(device)

    num_epochs = 20

    if(optimizer == None):
        optimizer = optim.SGD(net.parameters(), lr=5e-3, momentum=0, weight_decay=1e-5)

    if(scheduler == None):
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=(num_epochs/2)*len(train_loader), step_size_down=(num_epochs/2)*len(train_loader), mode='triangular')

    if isinstance (scheduler, lr_scheduler.ReduceLROnPlateau):
        sched_name = 'reduce'
    elif isinstance (scheduler, lr_scheduler.ReduceLROnPlateau):
        sched_name = 'exp'
    else:
        sched_name = 'cyclic'

    training_losses = []
    validation_losses = []
    # best_val_accuracy = 0  # Initialize the best validation accuracy

    # loop through epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        net.train()  # set to training mode

        lr_start = optimizer.param_groups[0]['lr'] # lr at start of epoch

        # loop through minibatches
        for i, data in enumerate(train_loader, 0):
            inputs, labels, path = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # set the accumulated gradients back to zero

            outputs = net(inputs) # pass data through the network

            loss = criterion(outputs, labels) # compute loss

            loss.backward() # do backpropagation and compute gradients
            optimizer.step() # update the parameters according to optimizer


            if(sched_name == 'cyclic'): # do a step if cyclic LR
                scheduler.step() 

            running_loss += loss.item()

        lr_end = optimizer.param_groups[0]['lr']

        # get training loss
        average_training_loss = running_loss / len(train_loader) # get len of training loss

        val_accuracy, average_val_loss = test_model(net, val_loader, device, criterion)

        if(split_songs):
            aggregate_val_accuracy = test_model_aggregate(net, val_loader, device)

        # Append the average losses to their respective lists
        training_losses.append(average_training_loss)
        validation_losses.append(average_val_loss)

        if(sched_name == 'reduce'):
            scheduler.step(average_training_loss)
        elif(sched_name == 'exp'):
            if((epoch+1) % 2 == 0):
                scheduler.step()

        # Save the model if the validation accuracy is the best so far
        # if val_accuracy > best_val_accuracy:
        #     best_val_accuracy = val_accuracy
        #     torch.save(net.state_dict(), f"models/{model_path}")  # Save the best model

        results_str = f"Epoch {epoch + 1} | LR: {lr_start:.2e} -> {lr_end:.2e} | Train Loss: {average_training_loss:.3f} | Val Loss: {average_val_loss:.3f} | Val Acc: {val_accuracy:.2f}%"

        if(split_songs):
            results_str +=  f" | Val Acc (agg): {aggregate_val_accuracy:.2f}%"

        print(results_str)

    torch.save(net.state_dict(), f"models/{model_path}")
    print("Training finished")
    return training_losses, validation_losses