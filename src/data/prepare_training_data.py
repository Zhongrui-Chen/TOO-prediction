from sklearn.model_selection import train_test_split
from src.data.quality_control import filter_dataset, interested_sites
from torch.utils.data import DataLoader
import numpy as np

def prepare_training_data(dataset_dict, q, k, batch_size=64, random_state=42):
    # Load the feature matrix
    feature_filepath = './data/interim/features/features_q={}_k={}.npy'.format(q, k)
    features = np.load(feature_filepath)
    y = []
    for tumour_id in filter_dataset(dataset_dict, q):
        ps = dataset_dict['primary_site_dict'][tumour_id]
        # y.append(dataset_dict['primary_sites'].index(ps))
        y.append(interested_sites.index(ps))
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=random_state)
    # Prepare the data loaders
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
    dataloaders_dict = {
        'train': DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True),
        'val': DataLoader(list(zip(X_valid, y_valid)), batch_size=batch_size, shuffle=True),
        'test': DataLoader(list(zip(X_test, y_test)))
    }
    return dataloaders_dict