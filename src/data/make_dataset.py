# -----------------------------------------------------------------
# Make the dataset by processing the COSMIC mutation TSV file
# Steps:
#   1. Filter and cleanse the dataframe
#   2. Make the dataset by constructing an aggregative dictionary
#   3. Pickle the dataset dict and store it in the disk
# -----------------------------------------------------------------

import pickle
import torch
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import argparse
from ast import literal_eval

def make_dataset(df_mut): # FIXME
    '''
    Make the dataset by constructing the data dictionary
    '''
    # lookup_table = get_cds_lookup_table()
    # loc_dict_mut = get_loc_dict(df_mut)

    sample_ids = set()
    # cmut_dict = defaultdict(list) # sample_id -> [(coding, mut_type, gene, cmut)]
    # gmut_dict = defaultdict(list) # sample_id -> [(mut_type, chrom, gpos)]
    snv_dict = defaultdict(list)
    site_dict = defaultdict(list) # sample_id -> site

    for row in tqdm(df_mut.itertuples(index=False), total=len(df_mut)):
        sample_ids.add(row.sample_id)
        site_dict[row.sample_id] = row.site
        snv_dict[row.sample_id].append((row.f5, (row.ref, row.alt), row.f3))
    
    # cnv_dict = defaultdict(list) # sample_id -> [(gene, total_cn, mut_type (loss or gain))]
    # for row in tqdm(df_cnv.itertuples(index=False), total=len(df_cnv)):
    #     cnv_dict[row.sample_id].append((row.gene, row.total_cn, row.mut_type))

    dataset = { # Sample mappings
        'sample_ids': list(sample_ids),
        'snv_dict': snv_dict,
        'site_dict': site_dict
    }

    # cnt_affected_samples = 0
    # cnt_affected_mutations = 0
    # for sample_id in dataset['sample_ids']:
    #     if before[sample_id] > after[sample_id]:
    #         cnt_affected_samples += 1
    #         cnt_affected_mutations += before[sample_id] - after[sample_id]
    #         # print('Before: {}, after: {}'.format(before[sample_id], after[sample_id]))

    # print('cnt_affected_samples: {}({}), cnt_affected_mutations: {}({})'.format(cnt_affected_samples, len(dataset['sample_ids']), cnt_affected_mutations, sum(before.values())))

    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Test with a fraction of rows', type=int)
    args = parser.parse_args()
    test_nrows = args.test
    processed_mut_filepath = './data/processed/ProcessedGenomeScreensMutantExport.tsv'
    processed_cnv_filepath = './data/processed/ProcessedCompleteCNA.tsv'

    print('Reading the processed mut file')
    df_mut = pd.read_csv(processed_mut_filepath, sep='\t', nrows=test_nrows)

    # print('Reading the processed CNV file')
    # df_cnv = pd.read_csv(processed_cnv_filepath, sep='\t', nrows=test_nrows)

    # Make the dataset
    print('Make the dataset')
    dataset = make_dataset(df_mut) # FIXME

    # print(dataset)

    # Store the dataset
    if not test_nrows:
        dataset_filepath = './data/interim/dataset.pkl'
        with open(dataset_filepath, 'wb') as f:
            pickle.dump(dataset, f)
        print('The dataset is stored in {}'.format(dataset_filepath))

if __name__ == '__main__':
    main()