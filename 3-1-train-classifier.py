import os
from random import sample
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from imblearn.over_sampling import SMOTE

from lib.utils import get_sites, normalized
from lib.models import BaseNet

def prepare_dataloaders(features, labels, batch_size):
    random_state = 42 # FIXME: for reproducible

    # Split the training set into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.1, random_state=random_state, shuffle=True, stratify=labels)

    # Resample the training set
    sampler = SMOTE(random_state=random_state, sampling_strategy='minority')
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    # Prepare the dataloaders
    dataloaders = {
        'train': DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True),
        'val': DataLoader(list(zip(X_valid, y_valid)), shuffle=True),
    }

    return dataloaders

def train_epoch(model, optimizer, train_loader, criterion):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss

# def cal_balanced_acc(y_true, y_pred):
#     # y_true = y_true.detach().cpu().numpy()
#     y_pred = y_pred.detach().cpu().numpy()
#     return balanced_accuracy_score(y_true, y_pred)

def val_epoch(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            # Calculate balanced accuracy
            for p in preds:
                y_pred.append(p.item())
            y_true.extend(labels.data)
            # corrects += torch.sum(y_pred == labels.data)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
    # acc = (corrects / len(val_loader.dataset)).item()

    acc   = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    val_loss = val_loss / len(val_loader.dataset)

    return acc, b_acc, val_loss

def report_to_terminal(epoch, num_epochs, train_loss, val_loss, acc, b_acc, updated=False):
    print('[Epoch {:2d}/{}] '.format(epoch + 1, num_epochs), end='')
    print('Train loss: {:.6f}'.format(train_loss), end=' | ')
    print('Val loss: {:.6f}'.format(val_loss), end = ' ')
    print('Accuracy: {:.6f} Balanced: {:.6f}{}'.format(acc, b_acc, ' [Updated]' if updated else ''))

def train_model(model, dataloaders):
    best_acc = -1
    n_epochs = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    n_unimproved_epochs = 0
    patience = 10

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, optimizer, dataloaders['train'], criterion)
        acc, b_acc, val_loss = val_epoch(model, dataloaders['val'], criterion)

        if b_acc > best_acc:
            n_unimproved_epochs = 0
            best_acc = b_acc
            report_to_terminal(epoch, n_epochs, train_loss, val_loss, acc, b_acc, updated=True)
            torch.save(model.state_dict(), os.path.join('./experiments/models/normal.pt'))
        else:
            n_unimproved_epochs += 1
            report_to_terminal(epoch, n_epochs, train_loss, val_loss, acc, b_acc)

            # if n_unimproved_epochs >= int(patience / 2):

            if n_unimproved_epochs >= patience:
                print('--- Early stopping ---')
                break

    print('Best Acc: {:.6f}'.format(best_acc))

def main():
    # torch.manual_seed(42) # FIXME
    sites = get_sites('f')
    src_path = './features'
    # Get the table
    tab = pd.read_csv('./data/ICGC-dataset/f-train-table.csv')
    # tab = tab.loc[tab['site'].map(lambda x: x in sites)]
    features = []
    labels   = []
    for r in tqdm(tab.itertuples(), total=len(tab)):
        feats = []
        # for feat_name in ['sbs-sig', 'sbs-rd']:
        # for feat_name in ['indel-sig', 'indel-rd']:
        for feat_name in ['cnv-sig', 'cnv-rd']:
            feat = np.load(os.path.join(src_path, feat_name, r.sample_id + '.npy'))
            feats.append(feat)
        feats = np.concatenate(feats)
        features.append(feats)
        labels.append(sites.index(r.site))

    features = np.array(features).astype(np.float32)
    dataloaders = prepare_dataloaders(features, labels, batch_size=32)

    in_size = features.shape[1]
    out_size = len(sites)

    model = BaseNet(in_size, out_size)
    train_model(model, dataloaders)

if __name__ == "__main__":
    main()