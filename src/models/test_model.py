# import pickle
import torch
import seaborn as sns
from matplotlib import pyplot as plt
# import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report, precision_recall_fscore_support
from src.models.networks.base import BaseNet
from src.utils.model_versioning import ModelConfigArgumentParser, get_name
from src.data.make_dataset import sites_of_interest

def get_site_label(idx):
    return sites_of_interest[idx]

def get_confusion_matrix(y_true, y_pred):
    true_labels = [sites_of_interest[y] for y in y_true]
    pred_labels = [sites_of_interest[y] for y in y_pred]
    return confusion_matrix(true_labels, pred_labels, labels=sites_of_interest, normalize='true') # Recall

def save_confusion_heatmap(cm, filepath):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax = sns.heatmap(cm, annot=True, cmap='Blues')
    ax.set_xticklabels(sites_of_interest, rotation=90)
    ax.set_yticklabels(sites_of_interest, rotation=0)
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
            y_true.append(y.item())
            y_pred.append(pred.item())
    return y_true, y_pred

def test_model(model, model_id, test_dataloader, device):
    # device = torch.device('mps' if model_config.mps else 'cpu')
    y_true, y_pred = inference(model, test_dataloader, device)
    # Generate the confusion matrix and save it as graph
    cm = get_confusion_matrix(y_true, y_pred)
    save_confusion_heatmap(cm, './reports/figures/confusion-heatmap-{}.png'.format(model_id))
    # Report the metrics
    print(classification_report(y_true, y_pred, target_names=sites_of_interest))
    print('The balanced accuracy is {}'.format(balanced_accuracy_score(y_true, y_pred)))

def main():
    # model_config = ModelConfigArgumentParser().get_config()
    device = torch.device('cpu')
    # model_name = model_config.get_model_name()
    model_id = '20220701-0452'

    # Load the testing data
    with open('./data/interim/test_data/' + model_id, 'rb') as f:
        test_data = torch.load(f)
    # Load the model
    in_size = 36480
    hidden_size = 256
    out_size = 13
    model = BaseNet(in_size, hidden_size, out_size)
    model.load_state_dict(torch.load('./models/' + get_name('model', model_id, 'pt')))
    
    # Test and report
    test_model(model, model_id, test_data, device)
    # print('Testing accuracy: {} ({} / {})'.format(accuracy, corrects, total))

if __name__ == '__main__':
    main()