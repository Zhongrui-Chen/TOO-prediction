import logging
import json
from random import random
import time
import copy
import pickle
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import torch
from src.models.networks.base import BaseNet
import torch.nn as nn
from src.utils.model_config import ModelConfigArgumentParser
import warnings

import torch.optim as optim
# from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR
# import argparse
from tqdm import tqdm
from src.data.prepare_training_data import prepare_training_data

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# from src.data.quality_control import filter_dataset

# def prepare_data(config, dataset_dict, q, features):
#     y = []
#     for tumour_id in filter_dataset(dataset_dict, q):
#         ps = dataset_dict['primary_site_dict'][tumour_id]
#         y.append(dataset_dict['primary_sites'].index(ps))
#     random_state = config['randomState'] if config['randomState'] else None
#     X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=random_state)
#     # Store the data for testing afterwards
#     test_data = {
#         'X': X_test,
#         'y': y_test
#     }
#     # Prepare the training data loaders
#     X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
#     batch_size = config['batchSize']
#     dataloaders_dict = {
#         'train': DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True, num_workers=2),
#         'val': DataLoader(list(zip(X_valid, y_valid)), batch_size=batch_size, shuffle=True, num_workers=2)
#     }
#     return dataloaders_dict, test_data

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # for epoch in tqdm(range(num_epochs)):
    for epoch in range(num_epochs):
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
            
            if phase == 'val':
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)
                epoch_balanced_acc = balanced_accuracy_score(running_trues, running_preds)
                if epoch_balanced_acc > best_acc:
                    best_acc = epoch_balanced_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                print('[Epoch {}/{}] '.format(epoch + 1, num_epochs), end='')
                    # print('-' * 10)
                print('Loss: {:.4f} Acc: {:.4f} Balanced Acc: {:.4f}'.format(epoch_loss, epoch_acc, epoch_balanced_acc))
        # print()
            
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('Best balanced Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    # return model, val_acc_history
    return model

def main():
    # Make training reproducible
    torch.manual_seed(42)

    # Get the configuration via the command line
    model_config = ModelConfigArgumentParser().get_config()
    model_name = model_config.get_model_name()
    device = torch.device('mps' if model_config.mps else 'cpu')

    # Initialize the model
    model = BaseNet(model_config.k, model_config.hidden_size)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=model_config.lr, momentum=0.9)
    # scheduler = ReduceLROnPlateau(optimizer, 'max')
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    # scheduler = StepLR(optimizer, step_size=512, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    # Load the config
    with open('./config.json', 'r') as f:
        config = json.load(f)
    # Load the dataset dict
    with open('./data/interim/dataset_dict.pkl', 'rb') as f:
        dataset_dict = pickle.load(f)
    # Load the feature matrix
    # features = np.load('./data/interim/features.npy')
    dataloaders_dict = prepare_training_data(dataset_dict, model_config.q, model_config.k)
    # Store the testing data
    with open('./data/interim/test_data/' + model_name, 'wb') as f:
        # pickle.dump(dataloaders_dict['test'], f)
        torch.save(dataloaders_dict['test'], f)
    model = train_model(model, dataloaders_dict, criterion, optimizer, device, num_epochs=model_config.num_epochs)
    # Save the trained model
    torch.save(model.state_dict(), './models/' + model_name)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()