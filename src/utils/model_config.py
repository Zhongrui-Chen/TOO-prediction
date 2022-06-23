import argparse

# class FeatureConfig:
#     def __init__(self, config_dict):
#         self.q = config_dict['q']
#         self.k = config_dict['k']
#     def get_feature_name(self):
#         return 'features_q={}_k={}.npy'.format(self.q, self.k)

class ModelConfig:
    # def __init__(self, q, k, lr, num_epochs, hidden_size):
    def __init__(self, config_dict):
        self.q = config_dict['q']
        self.k = config_dict['k']
        self.lr = config_dict['lr']
        self.num_epochs = config_dict['epochs']
        self.hidden_size = config_dict['hidden']
        self.mps = config_dict['mps']
    def get_model_name(self):
        return 'model_q={}_k={}_lr={}_epochs={}_hidden={}.pt'.format(self.q, self.k, self.lr, self.num_epochs, self.hidden_size)

class ModelConfigArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--q', help='Quality threshold', type=int)
        self.add_argument('--k', help='K of K-mers', type=int)
        self.add_argument('--lr', help='Learning rate', type=float)
        self.add_argument('--epochs', help='Number of epochs', type=int)
        self.add_argument('--hidden', help='Hidden size', type=int)
        self.add_argument('--mps', help='Option to enable M1 GPU support', action='store_true')  
    def get_config(self):
        config_dict = {
            # The default configuration
            'q': 75,
            'k': 3,
            'lr': 0.01,
            'epochs': 25,
            'hidden': 256,
            'mps': False
        }
        args = self.parse_args()
        for key, val in vars(args).items():
            if val is not None:
                config_dict[key] = val
        return ModelConfig(config_dict)