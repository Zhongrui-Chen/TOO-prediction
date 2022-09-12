import torch
from src.models.networks.autoencoder import Autoencoder
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

tensor_transform = transforms.ToTensor()

# Initialize the model and load the weights
filepath = './src/experiments/autoencoder/model.pt'
model = Autoencoder(784, 100)
weights = torch.load(filepath)
model.load_state_dict(weights)

# Test
test_dataset = datasets.MNIST(
        root="./data", train=False, transform=tensor_transform, download=True
    )

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=10, shuffle=False
)

with torch.no_grad():
    for batch_features in test_loader:
        batch_features = batch_features[0]
        test_examples = batch_features.view(-1, 784)
        encoded = model.encode(test_examples)
        reconstruction = model(test_examples)
        break

with torch.no_grad():
    number = 10
    plt.figure(figsize=(20, 10))
    for index in range(number):
        # display original
        ax = plt.subplot(3, number, index + 1)
        plt.imshow(test_examples[index].numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display encoded
        ax = plt.subplot(3, number, index + 1 + number)
        plt.imshow(encoded[index].numpy().reshape(10, 10))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, number, index + 1 + number * 2)
        plt.imshow(reconstruction[index].numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()