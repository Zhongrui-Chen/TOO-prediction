# import argparse
import time
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.networks.autoencoder import Autoencoder
from src.utils.classes_selection import get_interested_sites
from src.models.networks.base_fs import BaseNet, EncoderNet
# from src.models.networks.autoencoder import Autoencoder
from src.utils.model_versioning import ModelConfigArgumentParser, get_model_name
import torch.optim as optim
import pandas as pd
# import warnings; warnings.filterwarnings('ignore')
import os
from ray import tune
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
from ray.tune import with_parameters
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN
# from tqdm import tqdm

sites_of_interest = get_interested_sites()

# Load the features
def prepare_dataloaders(features, labels, batch_size, random_state=42): # FIXME: for reproducible
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_state, shuffle=True)
    # Split the training set into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, shuffle=True)
    # Resample the training set
    sampler = SMOTE(random_state=random_state)
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    # Prepare the dataloaders
    dataloaders = {
        'train': DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True),
        'val': DataLoader(list(zip(X_valid, y_valid)), batch_size=batch_size, shuffle=True),
        'test': DataLoader(list(zip(X_test, y_test)), shuffle=True)
    }

    return dataloaders

def train_epoch(model, optimizer, train_loader, criterion, device):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss

def val_epoch(model, val_loader, criterion, device):
    model.eval()
    corrects = 0
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
    acc = (corrects / len(val_loader.dataset)).item()
    val_loss = val_loss / len(val_loader.dataset)
    return acc, val_loss

def report_to_terminal(epoch, num_epochs, train_loss, val_loss, acc, updated=False):
    print('[Epoch {:2d}/{}] '.format(epoch + 1, num_epochs), end='')
    print('Train loss: {:.6f}'.format(train_loss), end=' | ')
    print('Val loss: {:.6f}'.format(val_loss), end = ' ')
    print('Accuracy: {:.6f}{}'.format(acc, ' [Updated]' if updated else ''))

def train_model(config, static_config, dataloaders, tune_flag=False, checkpoint_dir=None):
    # since = time.time()
    best_acc = -1

    # Initiate the model and hyperparameters according to the config
    in_size = static_config['in_size']
    out_size = len(sites_of_interest)
    model = BaseNet(in_size, config['hidden'], out_size)
    # ae = Autoencoder(9524, 300)
    # ae.load_state_dict(torch.load('./models/autoencoder-20220911-1745.pt')) # FIXME Hard-coded
    # model = EncoderNet(in_size, config['hidden'], out_size, ae)
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    criterion = nn.CrossEntropyLoss()
    device = torch.device(static_config['device'])

    # if checkpoint_dir:
    #     path = os.path.join(checkpoint_dir, "checkpoint")
    #     best_model_wts, optimizer_state = torch.load(path)
    #     optimizer.load_state_dict(optimizer_state)

    for epoch in range(static_config['epochs']):
        train_loss = train_epoch(model, optimizer, dataloaders['train'], criterion, device)
        acc, val_loss = val_epoch(model, dataloaders['val'], criterion, device)

        if acc > best_acc:
            best_acc = acc
            # Report current performance to Tune
            if tune_flag:
                tune.report(mean_accuracy=acc)
                torch.save(model.state_dict(), './model.pt')
            else:
                report_to_terminal(epoch, static_config['epochs'], train_loss, val_loss, acc, updated=True)
                torch.save(model.state_dict(), './models/model-' + static_config['id'] + '.pt')
        else:
            report_to_terminal(epoch, static_config['epochs'], train_loss, val_loss, acc)

        # Save the model
        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((best_model_wts, optimizer.state_dict()), checkpoint_dir)

        ## Print current performance to terminal
        # report_to_terminal(epoch, config['epochs'], train_loss, val_loss, acc)
            
    # time_elapsed = time.time() - since
    # print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # print('Best Acc: {:4f}'.format(best_acc))
    # model.load_state_dict(best_model_wts)

def main():
    # FIXME: Make training reproducible
    torch.manual_seed(42)

    # Get the configuration via the command line
    parser = ModelConfigArgumentParser()
    feature_codename = parser.get_feature_codename()
    static_config = parser.get_config()
    tune_flag = parser.get_tune_flag()

    if tune_flag is None:
        tune_flag = False

    # Load the feature matrix
    feature_dirpath = './data/interim/features/' + feature_codename + '/'
    samples_df = pd.read_csv('./data/interim/dataset/sites.tsv', sep='\t')
    cnv_sample_id_list_filepath = './data/processed/CNVSampleIDList.tsv'
    cnv_sample_ids = pd.read_csv(cnv_sample_id_list_filepath, sep='\t')['sample_id'].to_list()
    sample_ids = []
    sites = samples_df['site'].to_list()
    labels = []

    cnv_cnt = 0
    total_cnt = 0
    for idx, sid in enumerate(samples_df['sample_id'].to_list()):
        if sites[idx] in sites_of_interest:
            total_cnt += 1
            if sid in cnv_sample_ids:
                cnv_cnt += 1 
                sample_ids.append(sid)
                labels.append(sites_of_interest.index(sites[idx]))

    print('Total: {}, CNV: {}'.format(total_cnt, cnv_cnt))

    # in_size = 3198 # FIXME
    # Load the autoencoder
    # ae = Autoencoder(9524, 300)
    # ae.load_state_dict(torch.load('./models/autoencoder-20220910-1115.pt')) # FIXME Hard-coded
    features = []
    print('Loading the features...')
    for sid in tqdm(sample_ids):
        feat = np.load(feature_dirpath + str(sid) + '.npy')
        # Convert the feature vector from numpy array into tensor
        # feat_tensor = torch.tensor(feat, dtype=torch.float32)
        # Encode the feature vector
        # feat = ae.encode(feat_tensor)
        features.append(feat)
    features = np.squeeze(np.array(features, dtype=np.float32))
    # print('Feature matrix shape:', (len(features), features[0].size()))
    print('Feature matrix shape:', features.shape)

    # Dimensionality reduction
    # print('Dimensionality reducing...')
    # pca = KernelPCA(n_components=300, kernel='rbf')
    # # tsne = TSNE(n_components=3, learning_rate='auto', init='pca')
    # features = pca.fit_transform(features)
    # # features = tsne.fit_transform(features)
    # print('Feature matrix shape after:', features.shape)

    # for name, param in ae.named_parameters():
    #     print(name)
    # return
    # Apply the encoder to the feature vectors
    # print('Encoding...')
    # encoded_features = []
    # for feat in features:
        # feat = torch.from_numpy(feat)        
        # enc_feat = ae.encode(feat)
        # encoded_features.append(enc_feat.detach().numpy())
        
    # features = np.squeeze(np.array(encoded_features, dtype=np.float32))

    # print('Encoded feature matrix shape:', features.shape)

    static_config['in_size'] = 9524 # FIXME

    # Prepare the training & testing sets
    print('Preparing the data with batch size = {}...'.format(static_config['batch']))
    dataloaders = prepare_dataloaders(features, labels, static_config['batch']) # FIXME

    # Store the testing data
    test_filepath = './data/interim/test_data/' + static_config['id']
    with open(test_filepath, 'wb') as f:
        torch.save(dataloaders['test'], f)
    print('The testing data is stored in {}'.format(test_filepath))

    # Train the model
    print('Training the model...')
    initial_params = [
        {'lr': 0.1, 'momentum': 0.8, 'hidden': 700}
    ]

    if tune_flag:
        search_space = {
            'lr': tune.qloguniform(0.01, 0.1, 0.005),
            'momentum': tune.quniform(0.5, 0.95, 0.05),
            'hidden': tune.qrandint(512, 1024, 16)
        }

        # hyperopt_search = HyperOptSearch(search_space, metric="mean_accuracy", mode="max")
        algo = HyperOptSearch(points_to_evaluate=initial_params)
        algo = ConcurrencyLimiter(algo, max_concurrent=7)
        
        analysis = tune.run(
            with_parameters(train_model, static_config=static_config, dataloaders=dataloaders, tune_flag=True),
            config=search_space,
            num_samples=100,
            metric='mean_accuracy',
            mode='max',
            scheduler=ASHAScheduler(),
            search_alg=algo,
            keep_checkpoints_num=1,
            checkpoint_score_attr='mean_accuracy'
        )

        print("Best hyperparameters found were: ", analysis.best_config)
    else:
        train_model(config=initial_params[0], static_config=static_config, dataloaders=dataloaders)

if __name__ == '__main__':
    main()