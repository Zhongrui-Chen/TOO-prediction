import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE

from lib.utils import get_sites
from lib.models import MultiEncoderNet

def prepare_dataloaders(X_train, y_train, X_valid, y_valid, batch_size=32):
    # random_state = 42 # FIXME: for reproducible

    # Hold out a portion of data for testing
    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=random_state, shuffle=True, stratify=labels)

    # X_train = np.array(X_train)
    # X_test  = np.array(X_test)
    # y_train = np.array(y_train)
    # y_test  = np.array(y_test)

    # Load the training set

    # with open(os.path.join(exper_path, 'X-train'), 'rb') as f:
    #     X_train = np.load(f)

    # with open(os.path.join(exper_path, 'y-train'), 'rb') as f:
    #     y_train = np.load(f)

    # print(y_train)

    # Split the training set into training and validation sets
    # X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.1, random_state=random_state, shuffle=True, stratify=labels)

    # Resample the training set
    # sampler = SMOTE(random_state=random_state)
    # X_train, y_train = sampler.fit_resample(X_train, y_train)
    sampler = SMOTE(sampling_strategy='minority')
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    # Prepare the dataloaders
    dataloaders = {
        'train': DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True),
        'val': DataLoader(list(zip(X_valid, y_valid)), shuffle=True)
    }

    return dataloaders

# def prepare_dataloaders(X_train, X_val, y_train, y_val, batch_size=32):
#     # Resample the training set
#     sampler = SMOTE(sampling_strategy='minority', random_state=42)
#     X_train, y_train = sampler.fit_resample(X_train, y_train)
#     dataloaders = {
#         'train': DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True),
#         'val':   DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=True),
#     }
#     return dataloaders

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
    print('[Epoch {:2d}/{}] '.format(epoch, num_epochs), end='')
    print('Train loss: {:.6f}'.format(train_loss), end=' | ')
    print('Val loss: {:.6f}'.format(val_loss), end = ' ')
    print('Accuracy: {:.6f} Balanced: {:.6f}{}'.format(acc, b_acc, ' [Updated]' if updated else ''))

def train_model(model, dataloaders, split_idx):
    best_acc = -1
    n_epochs = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    unfreeze_flag = False
    patience = 10
        # for g in optimizer.param_groups:
        #     g['lr'] = g['lr'] * 0.5
        # for g in optimizer.param_groups:
        #     g['lr'] = 5e-4
    
        # for g in optimizer.param_groups:
        #     g['lr'] = g['lr'] * 0.2
        # n_unimproved_epochs = 0
    # if not unfreeze_flag:
    #     print('--- Unfreeze the pretrained layers ---')
    #     unfreeze_flag = True
    #     for enc in [model.mu_encoder, model.sbs_rd_encoder, model.indel_rd_encoder, model.cnv_rd_encoder, model.sbs_encoder, model.indel_encoder, model.cnv_encoder]:
    #         for param in enc.parameters():
    #             param.requires_grad = True
    #     n_unimproved_epochs = 0

    for epoch in range(n_epochs):
        epoch = epoch + 1
        train_loss = train_epoch(model, optimizer, dataloaders['train'], criterion)
        acc, b_acc, val_loss = val_epoch(model, dataloaders['val'], criterion)

        if b_acc > best_acc:
            n_unimproved_epochs = 0
            best_acc = b_acc
            report_to_terminal(epoch, n_epochs, train_loss, val_loss, acc, b_acc, updated=True)
            torch.save(model.state_dict(), os.path.join('./experiments', 'models', 'f-128', str(split_idx), '{}.pt'.format(epoch)))
        else:
            n_unimproved_epochs += 1
            report_to_terminal(epoch, n_epochs, train_loss, val_loss, acc, b_acc)

            if n_unimproved_epochs >= patience:
                
                # else:
                # print('--- Early stopping ---')
                # break
                print('--- Early stopping ---')
                break


        # if epoch == 10 and not unfreeze_flag:
        #     print('--- Unfreeze the pretrained layers ---')
        #     unfreeze_flag = True
        #     for enc in [model.mu_encoder, model.sbs_rd_encoder, model.indel_rd_encoder, model.cnv_rd_encoder, model.sbs_encoder, model.indel_encoder, model.cnv_encoder]:
        #         for param in enc.parameters():
        #             param.requires_grad = True
        #     for g in optimizer.param_groups:
        #         g['lr'] = g['lr'] * 0.5

    # print('Best Acc: {:.6f}'.format(best_acc))
    return best_acc

def main():

    # Get the IDs of the experiment and the pre-trained autoencoder
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--exper', required=True)
    # # parser.add_argument('--ae', required=True)
    # args = parser.parse_args()
    # exper_id = args.exper
    # # ae_id    = args.ae

    # # Create the dir of experiment
    # exper_path = './experiments/' + exper_id
    # if not os.path.isdir(exper_path):
    #     os.mkdir(exper_path)
    # else:
    #     raise RuntimeError('Experiment exists!')

    # torch.manual_seed(42) # FIXME
    sites = get_sites('f')
    src_path = './features'

    # # Get the table
    # ftab = pd.read_csv('./data/dataset/final-table.csv')
    # ftab = ftab.loc[ftab['site'].map(lambda s: s in sites)]

    # features = []
    # labels   = []

    # for r in ftab.itertuples():
    #     feats = []
    #     for feat_name in ['sbs-sig', 'sbs-rd', 'indel-sig', 'indel-rd', 'cnv-sig', 'cnv-rd']:
    #         feat = np.load(os.path.join(src_path, feat_name, r.sample_id + '.npy')).astype(np.float32)
    #         feats.append(feat)
    #     feats = np.concatenate(feats)
    #     features.append(feats)
    #     labels.append(sites.index(r.site))

    # features = np.array(features).astype(np.float32)
    # dataloaders = prepare_dataloaders(features, labels, batch_size=32)
    # X, X_test, y, y_test = prepare_data(features, labels)

    # Save the test data
    # test_path = os.path.join(exper_path, 'test-data')
    # test_dataloder = DataLoader(list(zip(X_test, y_test)), shuffle=True)
    # with open(test_path, 'wb') as f:
    #     torch.save(test_dataloder, f)

    # Load the training set
    tab = pd.read_csv('./data/ICGC-dataset/f-train-table.csv')
    features = []
    labels   = []
    for r in tqdm(tab.itertuples(), total=len(tab)):
        feats = []
        for feat_name in ['sbs-sig', 'indel-sig', 'cnv-sig', 'sbs-rd', 'indel-rd', 'cnv-rd']:
            feat = np.load(os.path.join(src_path, feat_name, r.sample_id + '.npy'))
            feats.append(feat)
        feats = np.concatenate(feats)
        features.append(feats)
        labels.append(sites.index(r.site))

    features = np.array(features).astype(np.float32)
    labels   = np.array(labels)

    # Load the weights of pre-trainde autoencoder
    # ae_state_dict = torch.load(os.path.join('./experiments/ae/ae-f-128', '30.pt')) # 25

    # Create splits for k-fold validation
    k = 10
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    best_acc   = -1
    for split_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        # Train for a split
        split_idx = split_idx + 1
        print('=============== Split {}/{} ==============='.format(split_idx, k))
        sp = os.path.join('./experiments', 'models/f-128', str(split_idx))
        if not os.path.isdir(sp):
            os.makedirs(sp)
        # Initialize the model
        out_size = len(sites)
        model = MultiEncoderNet(128, out_size)
        # Freeze the encoder part of the model
        # model_dict = model.state_dict()
        # enc_state_dict = {k: v for k, v in ae_state_dict.items() if k in model_dict}
        # model_dict.update(enc_state_dict)
        # model.load_state_dict(model_dict)
        # for enc in [model.mu_encoder, model.sbs_rd_encoder, model.indel_rd_encoder, model.cnv_rd_encoder, model.sbs_encoder, model.indel_encoder, model.cnv_encoder]:
        # for enc in [model.mu_encoder, model.sbs_encoder, model.indel_encoder, model.cnv_encoder]:
        #     for param in enc.parameters():
        #         param.requires_grad = False

        X_train, y_train, X_valid, y_valid = features[train_idx], labels[train_idx], features[val_idx], labels[val_idx]
        dataloaders = prepare_dataloaders(X_train, y_train, X_valid, y_valid)
        acc = train_model(model, dataloaders, split_idx)
        if acc > best_acc:
            best_acc   = acc
            best_split = split_idx
    print('The best split is {}'.format(best_split))

    # Initialize the model
    # out_size = len(sites)
    # model = MultiEncoderNet(out_size)
    # # Freeze the encoder part of the model
    # model_dict = model.state_dict()
    # enc_state_dict = {k: v for k, v in ae_state_dict.items() if k in model_dict}
    # model_dict.update(enc_state_dict)
    # model.load_state_dict(model_dict)
    # for enc in [model.mu_encoder, model.sbs_rd_encoder, model.indel_rd_encoder, model.cnv_rd_encoder, model.sbs_encoder, model.indel_encoder, model.cnv_encoder]:
    # for enc in [model.mu_encoder, model.sbs_rd_encoder, model.indel_rd_encoder, model.cnv_rd_encoder, model.sbs_encoder, model.indel_encoder, model.cnv_encoder]:
        # for param in enc.parameters():
            # param.requires_grad = False

if __name__ == "__main__":
    main()