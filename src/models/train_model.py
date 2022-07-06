import json
import time
import copy
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.make_dataset import sites_of_interest
from src.models.networks.base_fs import BaseNet
from src.utils.model_versioning import ModelConfigArgumentParser
import torch.optim as optim
# from tqdm import tqdm

# Load the dataset dict
with open('./data/interim/dataset.pkl', 'rb') as f: # FIXME
    dataset = pickle.load(f)

# Load the feature matrix
print('[Loading the features]')
feature_filepath = './data/interim/features/features.npy'
features = np.load(feature_filepath)

def prepare_dataloaders(batch_size, random_state=42): # FIXME: for reproducible
    y = []
    for sample_id in dataset['sample_ids']:
        site = dataset['site_dict'][sample_id]
        y.append(sites_of_interest.index(site))
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=random_state)
    # Prepare the data loaders
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
    dataloaders = {
        'train': DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True),
        'val': DataLoader(list(zip(X_valid, y_valid)), batch_size=batch_size, shuffle=True),
        'test': DataLoader(list(zip(X_test, y_test)))
    }
    return dataloaders

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_balanced_acc = 0.0
    
    # for epoch in tqdm(range(num_epochs)):

    # Early stopping
    patience = int(0.10 * num_epochs) + 1

    for epoch in range(num_epochs):
        
        if patience <= 0:
            print('[Early stopping]')
            break

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            running_trues = torch.Tensor()
            running_preds = torch.Tensor()
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # outputs = model(inputs.float())
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'val':
                    running_trues = torch.cat([running_trues, labels])
                    running_preds = torch.cat([running_preds, preds])

            # scheduler.step()

            if phase == 'train':
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                print('[Epoch {:3d}/{}] '.format(epoch + 1, num_epochs), end='')
                    # print('-' * 10)
                print('Training loss: {:.4f}'.format(epoch_loss), end=' | ')
            
            if phase == 'val':
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)
                epoch_balanced_acc = balanced_accuracy_score(running_trues, running_preds)
                # print('[Epoch {:2d}/{}] '.format(epoch + 1, num_epochs), end='')
                    # print('-' * 10)
                print('Val loss: {:.4f} Acc: {:.4f} Balanced Acc: {:.4f}'.format(epoch_loss, epoch_acc, epoch_balanced_acc), end='')
                # print('Loss: {:.4f} Acc: {:.4f} Balanced Acc: {:.4f}'.format(epoch_loss, epoch_acc, epoch_balanced_acc), end='')
                # if epoch_balanced_acc > best_balanced_acc:
                if epoch_acc > best_acc:
                    # best_balanced_acc = epoch_balanced_acc
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(' [updated]', end='')
                    patience = int(0.2 * num_epochs) + 1
                else:
                    patience -= 1
                print()
            
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # print('Best balanced Acc: {:4f}'.format(best_balanced_acc))
    print('Best Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    # return model, val_acc_history
    return model

def main():
    # FIXME: Make training reproducible
    torch.manual_seed(42)

    # Get the configuration via the command line
    model_config = ModelConfigArgumentParser().get_config()
    model_name = model_config.get_model_name()
    device = torch.device('mps' if model_config.mps else 'cpu')

    # Calculate the in_size and out_size
    in_size = np.product(features.shape[1:])
    out_size = len(sites_of_interest)

    # Initialize the model
    print('Initializing the model with in_size={}, hidden_size={}, out_size={}'.format(in_size, model_config.hidden_size, out_size))
    model = BaseNet(in_size, model_config.hidden_size, out_size)
    model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=model_config.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=model_config.lr)
    criterion = nn.CrossEntropyLoss()

    # Prepare the training & testing data
    print('[Preparing the data with batch size = {}]'.format(model_config.batch_size))
    dataloaders = prepare_dataloaders(model_config.batch_size)

    # Store the testing data
    test_filepath = './data/interim/test_data/' + model_config.model_id
    with open(test_filepath, 'wb') as f:
        torch.save(dataloaders['test'], f)
    print('The testing data is stored in {}'.format(test_filepath))

    # Train the model
    print('[Training the model]')
    model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=model_config.num_epochs)

    # Save the trained model
    model_filepath = './models/' + model_name
    torch.save(model.state_dict(), model_filepath)
    print('The model is stored in {}'.format(model_filepath))

if __name__ == '__main__':
    main()