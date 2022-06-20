# -*- coding: utf-8 -*-
# import click
# import json
# import logging
# import pickle

def filter_dataset(dataset_dict, threshold):
    quality_ids = []
    for tumour_id in dataset_dict['tumour_ids']:
        count = 0
        for gene in dataset_dict['genes']:
            if (tumour_id, gene) in dataset_dict['mut_dict']:
                count += 1
        if count >= threshold:
            quality_ids.append(tumour_id)
    return quality_ids

# def main():
#     logger = logging.getLogger(__name__)
#     with open('./data/interim/dataset_dict.pkl', 'rb') as f:
#         dataset_dict = pickle.load(f)
#     with open('./config.json', 'r') as f:
#         config = json.load(f)
#     dataset_dict['qualified_tumour_ids'] = filter_dataset(dataset_dict, config['threshold'])
#     logger.info('The number of qualified tumour samples is {}, out of {}.'.format(len(dataset_dict['qualified_tumour_ids']), len(dataset_dict['tumour_ids'])))
#     with open('./data/interim/dataset_dict.pkl', 'wb') as f:
#         pickle.dump(dataset_dict, f)

# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     main()