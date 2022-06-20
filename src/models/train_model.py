import logging
import json
from random import random
import time
import copy
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from src.models.networks.base import BaseNet
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
from src.data.quality_control import filter_dataset

def prepare_data(config, dataset_dict, q, features):
    y = []
    for tumour_id in filter_dataset(dataset_dict, q):
        ps = dataset_dict['primary_site_dict'][tumour_id]
        y.append(dataset_dict['primary_sites'].index(ps))
    random_state = config['randomState'] if config['randomState'] else None
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=random_state)
    # Store the data for testing afterwards
    test_data = {
        'X': X_test,
        'y': y_test
    }
    # Prepare the training data loaders
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
    batch_size = config['batchSize']
    dataloaders_dict = {
        'train': DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(list(zip(X_valid, y_valid)), batch_size=batch_size, shuffle=True, num_workers=2)
    }
    return dataloaders_dict, test_data

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in tqdm(range(num_epochs)):
        # print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # outputs = model(inputs.float())
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)
            
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # print()
            
    time_elapsed = time.time() - since
    # print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    # return model, val_acc_history
    return model

def main():
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('features_filepath', help='The file path to the feature matrix', type=str)
    argparser.add_argument('--q', help='Quality threshold', type=int)
    argparser.add_argument('--k', help='K of K-mers', type=int)
    argparser.add_argument('--mps', help='Option to enable M1 GPU support')
    args = argparser.parse_args()

    # Initialize the model
    model = BaseNet()
    device = torch.device('mps' if args.mps else 'cpu')
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Model name
    k = args.k if args.k else 3
    q = args.q if args.q else 10
    model_name = 'model_q={}_k={}.pt'.format(q, k)
    # Load the config
    with open('./config.json', 'r') as f:
        config = json.load(f)
    # Load the dataset dict
    with open('./data/interim/dataset_dict.pkl', 'rb') as f:
        dataset_dict = pickle.load(f)
    # Load the feature matrix
    # features = np.load('./data/interim/features.npy')
    features_filepath = './data/interim/features/features_q={}_k={}.npy'.format(q, k)
    features = np.load(features_filepath)
    dataloaders_dict, test_data = prepare_data(config, dataset_dict, q, features)
    # Store the testing data
    with open('./data/interim/test_data/' + model_name + '.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    model = train_model(model, dataloaders_dict, criterion, optimizer, device, num_epochs=10)
    torch.save(model.state_dict(), './models/' + model_name)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()