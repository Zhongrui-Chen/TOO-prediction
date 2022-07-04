import argparse
from datetime import datetime

def get_time():
    return datetime.now().strftime("%Y%m%d-%H%M")

def get_name(prefix, id, suffix=''):
    return prefix + '-' + id + ('.' + suffix if len(suffix) > 0 else '')

class ModelConfig:
    def __init__(self, config_dict):
        self.model_id = config_dict['id']
        self.batch_size = config_dict['batch']
        self.lr = config_dict['lr']
        self.num_epochs = config_dict['epochs']
        self.hidden_size = config_dict['hidden']
        self.mps = config_dict['mps']
    def get_model_name(self):
        return get_name('model', self.model_id, 'pt')

class ModelConfigArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--id', help='Specify the model id', type=str)
        self.add_argument('--batch', help='The size of a mini-batch', type=int)
        self.add_argument('--lr', help='Learning rate', type=float)
        self.add_argument('--epochs', help='Number of epochs', type=int)
        self.add_argument('--hidden', help='Hidden size', type=int)
        self.add_argument('--mps', help='Option to enable M1 GPU support', action='store_true')  
    def get_config(self):
        config_dict = { # The default configuration
            'id': get_time(),
            'batch': 32,
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