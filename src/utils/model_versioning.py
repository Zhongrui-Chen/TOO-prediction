import argparse
from datetime import datetime

def get_time():
    return datetime.now().strftime("%Y%m%d-%H%M")

def get_name(prefix, id, suffix=''):
    return prefix + '-' + id + ('.' + suffix if len(suffix) > 0 else '')

def get_model_name(model_id):
    return get_name('model', model_id, 'pt')

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
        self.add_argument('--epochs', help='Number of epochs', type=int)
        self.add_argument('--device', help='Option to enable GPU support')

        self.add_argument('--feature', help='feature codename')

        # --- Tuneable parameters ---
        self.add_argument('--lr', help='Learning rate', type=float)
        self.add_argument('--momentum', help='Momentum of SGD', type=float)
        self.add_argument('--hidden', help='Hidden size', type=int)
        # --- Tuneable parameters ---        

        # --- Flags ---
        self.add_argument('--tune', help='The option to tune the hyper-parameters', action='store_true')
        # --- Flags ---

    def get_feature_codename(self):
        return self.parse_args().feature

    def get_tune_flag(self):
        tune_flag = self.parse_args().tune
        if tune_flag is None:
            return False
        return tune_flag

    def get_config(self):
        config_dict = { # The default configuration
            'id': get_time(),
            'batch': 32,
            # 'lr': 0.01,
            # 'momentum': 0.9,
            'epochs': 25,
            # 'hidden': 256,
            'device': 'cpu'
        }
        args = self.parse_args()
        for key, val in vars(args).items():
            if val is not None:
                config_dict[key] = val
        return config_dict