import pickle
import json
import argparse
from src.models.networks.base import BaseNet
from src.models.train_model import prepare_data, train_model
import torch
import torch.optim as optim
import torch.nn as nn
# from datetime import datetime
from src.data.quality_control import filter_dataset
from src.features.build_features import generate_feature_npy
from src.models.test_model import test_model
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mps', help='Option to enable M1 GPU support', action='store_true')
    args = argparser.parse_args()

    k = 3

    device = torch.device('mps' if args.mps else 'cpu')
    criterion = nn.CrossEntropyLoss()
    with open('./config.json', 'r') as f:
        config = json.load(f)
    # Load the dataset dictionary
    with open('./data/interim/dataset_dict.pkl', 'rb') as f:
        dataset_dict = pickle.load(f)

    threshold_range = [5, 10, 15, 20, 25, 50, 75, 100]
    accuracy_history = []
    best_accuracy = -1

    for q in threshold_range:
        model_name = 'model_q={}_k={}.pt'.format(q, k)

        # Load the features if possible
        features_filepath = './data/interim/features/features_q={}_k={}.npy'.format(q, k)
        try:
            features = np.load(features_filepath)
            print('The features {} are loaded'.format(features_filepath))
        except:
            # Build features
            features_filepath = generate_feature_npy(dataset_dict, q, k)
            features = np.load(features_filepath)

        # Prepare the data loaders and testing data
        tumour_ids = filter_dataset(dataset_dict, q)
        dataloaders_dict, test_data = prepare_data(config, dataset_dict, tumour_ids, features)

        # Try to load the trained model
        try:
            model = BaseNet()
            model.load_state_dict(torch.load('./models/' + model_name))
            print('The model {} is loaded'.format(model_name))
        except:
            net = BaseNet()
            net.to(device)
            optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
            # Train the model using the data with the threshold value
            print('Training the model for data with q = {}, k = {}'.format(q, k))
            model = train_model(net, dataloaders_dict, criterion, optimizer, device, num_epochs=10)
            torch.save(model.state_dict(), './models/' + model_name)
        
        # Evaluate the model
        corrects, total, accuracy = test_model(model, test_data, device)
        accuracy_history.append(accuracy)
        print('Testing accuracy: {} ({} out of {}) with q = {}'.format(accuracy, corrects, total, q))
        if accuracy > best_accuracy:
            best_q = q
            best_accuracy = accuracy

    print('The best threshold is {}, with accuracy = {}'.format(best_q, best_accuracy))
        
    # Draw the graph and save
    fig, ax = plt.subplots()
    ax.plot(threshold_range, accuracy_history)
    fig.savefig('./reports/figures/quality_threshold_search.png')
    plt.close(fig)

if __name__ == '__main__':
    main()