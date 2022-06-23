# -*- coding: utf-8 -*-
# import click
# import json
# import logging
# import pickle

interested_sites = [
    'kidney',
    'skin',
    'liver',
    'breast',
    'ovary',
    'haematopoietic_and_lymphoid_tissue',
    'prostate',
    'pancreas',
    'central_nervous_system',
    'lung',
    'oesophagus',
    'thyroid',
    'bone'
]
# interested_sites = ['skin',
#     'large_intestine',
#     'stomach',
#     'endometrium',
#     'lung',
#     'thyroid',
#     'soft_tissue'
# ]

def filter_by_sites(dataset_dict, tumour_ids):
    return [tid for tid in tumour_ids if dataset_dict['primary_site_dict'][tid] in interested_sites]

def filter_by_quality(dataset_dict, tumour_ids, q):
    ids = []
    for tid in tumour_ids:
        count = 0
        for gene in dataset_dict['genes']:
            if (tid, gene) in dataset_dict['mut_dict']:
                count += 1
        if count >= q:
            ids.append(tid)
    return ids

def filter_dataset(dataset_dict, q):
    ids = dataset_dict['tumour_ids']
    ids = filter_by_quality(dataset_dict, ids, q)
    ids = filter_by_sites(dataset_dict, ids)
    return ids

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