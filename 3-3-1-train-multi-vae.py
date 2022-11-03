import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from math import inf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from lib.utils import get_sites
from lib.models import MultiVAE

def prepare_dataloaders(features, labels):
    # random_state = 42 # FIXME: for reproducible

    # Split the training set into training and validation sets
    # X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.1, random_state=random_state, shuffle=True, stratify=labels)

    # # Resample the training set
    # # sampler = SMOTE(random_state=random_state)
    # # X_train, y_train = sampler.fit_resample(X_train, y_train)

    # # Prepare the dataloaders
    # dataloaders = {
    #     'train': DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True),
    #     'val': DataLoader(list(zip(X_valid, y_valid)), shuffle=True),
    # }

    dl = DataLoader(list(zip(features, labels)), batch_size=32, shuffle=True)

    return dl

def cal_kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())

def train_epoch(model, optimizer, train_loader, criterion):
    model.train()
    train_bce_loss = 0
    train_kl_loss  = 0
    for inputs, _ in train_loader:
        optimizer.zero_grad()
        # fus_inputs, fus_outputs, final_outputs, mu, logvar = model(inputs)
        final_outputs, mu, logvar = model(inputs)
        # inputs = torch.concat(inputs, dim=1)
        # fus_bl = criterion(fus_outputs, fus_inputs)
        final_bl = criterion(final_outputs, inputs)
        # bl = fus_bl + final_bl
        bl = final_bl
        kl = cal_kl_loss(mu, logvar)
        tl = bl + kl
        tl.backward()
        optimizer.step()
        train_bce_loss += bl.item()
        train_kl_loss  += kl.item()
        # train_loss += loss.item() * inputs.size(0)
    train_bce_loss = train_bce_loss / len(train_loader.dataset)
    train_kl_loss  = train_kl_loss / len(train_loader.dataset)
    total_train_loss = train_bce_loss + train_kl_loss
    return train_bce_loss, train_kl_loss, total_train_loss

def report_to_terminal(epoch, num_epochs, bl, kl, tl, updated=False):
    print('[Epoch {:2d}/{}] '.format(epoch + 1, num_epochs), end='')
    print('Loss: b = {:.6f}, k = {:.6f}, tl = {:.6f}'.format(bl, kl, tl), end='')
    print(' [Updated]' if updated else '')

def train_model(model, dataloader):
    min_loss = inf
    n_epochs = 50

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss(reduction='sum')

    n_unimproved_epochs = 0
    patience = 5

    for epoch in range(n_epochs):
        bl, kl, tl = train_epoch(model, optimizer, dataloader, criterion)

        if tl < min_loss:
            n_unimproved_epochs = 0
            min_loss = tl
            report_to_terminal(epoch, n_epochs, bl, kl, tl, updated=True)
            torch.save(model.state_dict(), os.path.join('experiments/ae/ae-f-128', '{}.pt'.format(epoch + 1)))
        else:
            n_unimproved_epochs += 1
            report_to_terminal(epoch, n_epochs, bl, kl, tl)

            if n_unimproved_epochs >= patience:
                print('--- Early stopping ---')
                break

def main():
    torch.manual_seed(42) # FIXME
    sites = get_sites('f')
    src_path = './features'
    
    # Get the IDs of the experiment and the pre-trained autoencoder
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--exper', required=True)
    # # parser.add_argument('--ae', required=True)
    # args = parser.parse_args()
    # exper_id = args.exper
    # exper_path = './experiments/' + exper_id

    # Get the table
    # stab = pd.read_csv('./data/ICGC-processed/simple-table.csv')
    # ctab = pd.read_csv('./data/ICGC-processed/cnv-table.csv')
    ftab = pd.read_csv('./data/ICGC-dataset/f-train-table.csv')

    # id_set = set(stab['sample_id']).intersection(set(ctab['sample_id']))

    # print(len(id_set))

    # ftab = ftab.loc[ftab['site'].map(lambda s: s in sites)]
    # stab = stab.loc[stab['sample_id'].map(lambda sid: sid in id_set)]

    features = []
    labels   = []

    for r in tqdm(ftab.itertuples(), total=len(ftab)):
        feats = []
        for feat_name in ['sbs-sig', 'sbs-rd', 'indel-sig', 'indel-rd', 'cnv-sig', 'cnv-rd']:
            feat = np.load(os.path.join(src_path, feat_name, r.sample_id + '.npy')).astype(np.float32)
            feats.append(feat)
        feats = np.concatenate(feats)
        features.append(feats)
        labels.append(sites.index(r.site))

    features = np.array(features).astype(np.float32)

    dataloader = prepare_dataloaders(features, labels)
    # print(len(features), len(features[0]), len(features[0][0]))

    # in_size = features.shape[1]
    model = MultiVAE(128)
    train_model(model, dataloader)

if __name__ == "__main__":
    main()