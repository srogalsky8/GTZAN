from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def custom_transform(image):
    # Crop the image to the desired region of interest (ROI)
    image = image.crop((55, 35, 390, 253))
    # Convert the cropped image to a PyTorch tensor
    return transforms.ToTensor()(image)

def load_data():
    # Define your data transformation (without resizing)
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(custom_transform),  # Apply the custom transformation
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    class CustomImageDataset(datasets.ImageFolder):
        def __init__(self, root, transform=None):
            super(CustomImageDataset, self).__init__(root=root, transform=transform)

    # Define the path to your data folder
    data_dir = 'data/images_original'

    # Create an instance of your custom dataset
    custom_dataset = CustomImageDataset(root=data_dir, transform=data_transform)

    # Calculate the size of the training and testing sets
    total_size = len(custom_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])
    batch_size = 64  # You can adjust this batch size as needed

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
