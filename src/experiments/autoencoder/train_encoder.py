import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from src.models.networks.autoencoder import Autoencoder
from tqdm import tqdm

# Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()
 
# Download the MNIST Dataset
dataset = datasets.MNIST(root = "./data",
                         train = True,
                        #  download = True,
                         transform = tensor_transform)
 
# DataLoader is used to load the dataset
# for training
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = 32,
                                     shuffle = True)

def main():
    model = Autoencoder(784, 100)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-3,
                             weight_decay = 1e-8)

    epochs = 5

    for epoch in range(epochs):
        loss = 0

        for batch_features, _ in loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 784)
            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch_features)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(loader)
        # display the epoch training loss
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))
    
    # Save the trained auto-encoder
    filepath = './src/experiments/autoencoder/model.pt'
    torch.save(model.state_dict(), filepath)

if __name__ == '__main__':
    main()