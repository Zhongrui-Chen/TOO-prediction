import pickle
import torch
import argparse
from src.models.networks.base import BaseNet

def test_model(model, test_data, device, k=3):
    model = BaseNet(k)
    model.load_state_dict(model.state_dict())
    model.to(device)
    model.eval()
    X_test = test_data['X']
    y_test = test_data['y']
    # Inference
    corrects = 0
    testset = list(zip(X_test, y_test))
    with torch.no_grad():
        for X, y in testset:
            X = torch.Tensor(X).to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            corrects += (int(predicted) == y)
    return corrects, len(testset), corrects / len(testset)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mps', help='Option to enable M1 GPU support', action='store_true')
    argparser.add_argument('model_name', help='Specify the model name', type=str)
    args = argparser.parse_args()
    model_name = args.model_name
    device = torch.device('mps' if args.mps else 'cpu')

    # Load the model
    model = BaseNet()
    model.load_state_dict(torch.load('./models/' + model_name))
    
    # Load the testing data
    with open('./data/interim/test_data/model_q=70_k=3.pt.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    # Test and report
    corrects, total, accuracy = test_model(model, test_data, device)
    print('Testing accuracy: {} ({} / {})'.format(accuracy, corrects, total))

if __name__ == '__main__':
    main()