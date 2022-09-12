from math import inf
import json
from pickletools import optimize
import random
import time
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from data.preprocess_mut import sites_of_interest
from src.utils.classes_selection import get_interested_sites
# from src.models.networks.base_fs import BaseNet
from src.models.networks.autoencoder import Autoencoder
from src.utils.model_versioning import ModelConfigArgumentParser
import torch.optim as optim
import pandas as pd

sites_of_interest = get_interested_sites()

# # Load the feature matrix
# print('Loading the features')
# feature_dirpath = './data/interim/features/feat_vectors/'
# samples_df = pd.read_csv('./data/interim/dataset/sites.tsv', sep='\t')
# sample_ids = samples_df['sample_id'].to_list()

# # Load the features
# features = []
# for sid in sample_ids:
#     fmap = np.load(feature_dirpath + str(sid) + '.npy')
#     features.append(fmap)
# features = np.array(features, dtype=np.float32)

def prepare_dataloader(features, labels, batch_size=32): # FIXME: for reproducible
    dataloader = DataLoader(list(zip(features, labels)), batch_size=batch_size, shuffle=True)
    return dataloader

def train_autoencoder(static_config, dataloader):
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    minimum_error = inf

    model = Autoencoder(in_size=static_config['in_size'], embed_size=static_config['embed_size'])
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)
    criterion = nn.MSELoss()
    device = torch.device(static_config['device'])
    num_epochs = static_config['epochs']

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, _ in dataloader:
            # inputs = inputs.view(-1, static_config['in_size']).to(device)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print('[Epoch {:2d}/{}] '.format(epoch + 1, num_epochs), end='')
        print('Training loss: {:.8f}'.format(epoch_loss), end='')
        if epoch_loss < minimum_error:
            minimum_error = epoch_loss
            print(' [updated]', end='')
            torch.save(model.state_dict(), './models/autoencoder-' + static_config['id'] + '.pt')
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

def main():
    # FIXME: Make training reproducible
    torch.manual_seed(42)

    # Get the configuration via the command line
    parser = ModelConfigArgumentParser()
    static_config = parser.get_config()

    # Load the features
    feature_codename = parser.get_feature_codename()
    print('Loading the features...')
    feature_dirpath = './data/interim/features/' + feature_codename + '/'
    samples_df = pd.read_csv('./data/interim/dataset/sites.tsv', sep='\t')
    cnv_sample_id_list_filepath = './data/processed/CNVSampleIDList.tsv'
    cnv_sample_ids = pd.read_csv(cnv_sample_id_list_filepath, sep='\t')['sample_id'].to_list()
    sample_ids = []
    sites = samples_df['site'].to_list()
    labels = []
    for idx, sid in enumerate(samples_df['sample_id'].to_list()):
        if sites[idx] in sites_of_interest and sid in cnv_sample_ids:
            sample_ids.append(sid)
            labels.append(sites_of_interest.index(sites[idx]))

    features = []
    for sid in sample_ids:
        feat = np.load(feature_dirpath + str(sid) + '.npy')
        features.append(feat)

    features = np.squeeze(np.array(features, dtype=np.float32))
    print('Feature matrix shape:', features.shape)

    # Calculate the in_size and embed_size
    static_config['in_size'] = np.product(features.shape[1:])
    static_config['embed_size'] = 300 # FIXME: hard-coded

    # Initialize the model
    # print('Initializing the model with in_size={}, embed_size={}'.format(in_size, embed_size))
    # model = Autoencoder(in_size, embed_size)
    # model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=parser.get_lr(), momentum=0.9)
    # criterion = nn.MSELoss()

    # Prepare the data for representation training
    dataloader = prepare_dataloader(features, labels)

    # Train the autoencoder
    print('Training the model...')
    train_autoencoder(static_config, dataloader)

    # # Save the trained encoder
    # encoder_filepath = './models/autoencoder-{}.pt'.format(static_config.id)
    # torch.save(model.state_dict(), encoder_filepath)
    # print('The auto-encoder model is stored in {}'.format(encoder_filepath))

    # # Visualize some vectors
    # k = 10
    # random_samples = random.sample(sample_ids, k)
    # with torch.no_grad():
    #     plt.figure(figsize=(20, 40))
    #     for sid in random_samples:
    #         # display original
    #         ax = plt.subplot(0, index + 1)
    #         plt.imshow(test_examples[index].numpy().reshape(28, 28))
    #         plt.gray()
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)

    #         # display encoded
    #         ax = plt.subplot(3, k, index + 1 + number)
    #         plt.imshow(encoded[index].numpy().reshape(10, 10))
    #         plt.gray()
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)

    #         # display reconstruction
    #         ax = plt.subplot(3, k, index + 1 + number * 2)
    #         plt.imshow(reconstruction[index].numpy().reshape(28, 28))
    #         plt.gray()
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #     plt.show()
    

if __name__ == '__main__':
    main()