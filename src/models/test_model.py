# import pickle
import torch
import seaborn as sns
from matplotlib import pyplot as plt
# import argparse
# from src.data.quality_control import interested_sites
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report, precision_recall_fscore_support
from src.models.networks.base import BaseNet
from src.utils.model_config import ModelConfigArgumentParser
from src.data.quality_control import interested_sites

def get_site_label(idx):
    return interested_sites[idx]

def get_confusion_matrix(y_true, y_pred):
    true_labels = [interested_sites[y] for y in y_true]
    pred_labels = [interested_sites[y] for y in y_pred]
    return confusion_matrix(true_labels, pred_labels, labels=interested_sites, normalize='true') # Recall

def save_confusion_heatmap(cm, filepath):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax = sns.heatmap(cm, annot=True, cmap='Blues')
    ax.set_xticklabels(interested_sites, rotation=90)
    ax.set_yticklabels(interested_sites, rotation=0)
    fig.savefig(filepath)
    print('The confusion matrix is saved to {}'.format(filepath))
    plt.close(fig)

def inference(model, test_dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            pred = torch.argmax(outputs, 1)
            # true_pred_pairs.append((y.item(), pred.item()))
            y_true.append(y.item())
            y_pred.append(pred.item())
    return y_true, y_pred

def test_model(model, test_dataloader, q, k, device):
    y_true, y_pred = inference(model, test_dataloader, device)
    # Generate the confusion matrix and save it as graph
    cm = get_confusion_matrix(y_true, y_pred)
    save_confusion_heatmap(cm, './reports/figures/confusion-heatmap-q={}-k={}.png'.format(q, k))
    # Report the metrics
    print(classification_report(y_true, y_pred, target_names=interested_sites))
    print('The balanced accuracy is {}'.format(balanced_accuracy_score(y_true, y_pred)))

def main():
    model_config = ModelConfigArgumentParser().get_config()
    device = torch.device('mps' if model_config.mps else 'cpu')
    model_name = model_config.get_model_name()

    # Load the model
    model = BaseNet(model_config.k, model_config.hidden_size)
    model.load_state_dict(torch.load('./models/' + model_name))
    
    # Load the testing data
    with open('./data/interim/test_data/' + model_name, 'rb') as f:
        test_data = torch.load(f)
    
    # Test and report
    test_model(model, test_data, model_config.q, model_config.k, device)
    # print('Testing accuracy: {} ({} / {})'.format(accuracy, corrects, total))

if __name__ == '__main__':
    main()