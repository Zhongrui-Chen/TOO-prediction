# -*- coding: utf-8 -*-
# import click
import logging
import json
import pickle
# from dotenv import find_dotenv, load_dotenv

# import pysam
from tqdm import tqdm
import pandas as pd
import hgvs.parser

# from src.utils import seq
from src.utils.uhgvs import is_hgvs_valid

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())

raw_filepath = './data/raw/'
ext_filepath = './data/external/'
interim_filepath = './data/interim/'
processed_filepath = './data/processed/'
logger = logging.getLogger(__name__)

def idcs_of_rows_with_isan_column(df, column):
    nan_map = df[column].isnull()
    isan_idcs = [idx for idx in range(len(df)) if not nan_map.iloc[idx]]
    return isan_idcs

def idcs_of_rows_with_valid_hgvs(df, hp, mut_type):
    valid_hgvs_idcs = []
    for idx in tqdm(range(len(df))):
        if is_hgvs_valid(str(df[mut_type].iloc[idx]), hp):
            valid_hgvs_idcs.append(idx)
    return valid_hgvs_idcs

def data_cleansing(df, cols_of_interest, hparser, hgvs_type):
    # Select interested columns
    df = df[cols_of_interest]
    # Remove rows with NaN important column values
    len_before_removing_nan = len(df)
    isan_idcs = idcs_of_rows_with_isan_column(df, hgvs_type)
    df = df.iloc[isan_idcs, :]
    logger.info('{} NaN rows were removed'.format(len_before_removing_nan - len(df)))
    # Remove duplicates
    len_before_removing_duplicates = len(df)
    df = df.drop_duplicates(subset=['ID_tumour', hgvs_type])
    logger.info('{} duplicate rows were removed'.format(len_before_removing_duplicates - len(df)))
    # Remove rows with invalid HGVSC description
    len_before_removing_invalid_hgvs = len(df)
    logger.info('Removing rows with invalid HGVS description')
    valid_hgvs_idcs = idcs_of_rows_with_valid_hgvs(df, hparser, hgvs_type)
    df = df.iloc[valid_hgvs_idcs, :]
    logger.info('{} rows with invalid HGVS description were removed'.format(len_before_removing_invalid_hgvs - len(df)))
    logger.info('The final number of samples is {}'.format(len(df)))
    return df

def get_summary_info(df):
    all_tumour_ids = list(set(df['ID_tumour']))
    logger.info('There are {} tumours in total'.format(len(all_tumour_ids)))
    all_genes = list(set(df['Gene name']))
    logger.info('There are {} genes in total'.format(len(all_genes)))
    all_primary_sites = list(set(df['Primary site']))
    logger.info('There are {} primary sites in total'.format(len(all_primary_sites)))
    with open('./data/interim/primary_site_list.txt', 'w') as f:
        for ps in all_primary_sites:
            f.write(ps + '\n')
    logger.info('The list of primary sites is written to {}'.format('./data/interim/primary_site_list.txt'))
    return all_tumour_ids, all_genes, all_primary_sites

def get_data_dictionaries(df, hgvs_type, cols_of_interest):
    # Construct the dictionaries of samples
    loc_of = {}
    
    # min_nb_of_mutated_genes = np.inf
    # max_nb_of_mutated_genes = -np.inf
    for col in cols_of_interest:
        loc_of[col] = df.columns.get_loc(col)
    mut_dict = {} # (tumour_id, gene) -> [mutations]
    primary_site_dict = {} # tumour_id -> primary_site
    for row in df.itertuples(index=False):
        tumour_id = row[loc_of['ID_tumour']]
        gene = row[loc_of['Gene name']]
        mut = row[loc_of[hgvs_type]]
        primary_site = row[loc_of['Primary site']]
        if tumour_id not in primary_site_dict:
            primary_site_dict[tumour_id] = primary_site
        if (tumour_id, gene) not in mut_dict:
            mut_dict[(tumour_id, gene)] = [mut]
        else:
            mut_dict[(tumour_id, gene)].append(mut)
    # for tumour_id in all_tumour_ids:
    #     count = 0
    #     for gene in all_genes:
    #         if (tumour_id, gene) in mut_dict:
    #             count += 1
    #     if count < min_nb_of_mutated_genes:
    #         min_nb_of_mutated_genes = count
    #     if count > max_nb_of_mutated_genes:
    #         max_nb_of_mutated_genes = count
    # print('The minimum and maximum number of mutated genes in a tumour are {} and {}.'.format(min_nb_of_mutated_genes, max_nb_of_mutated_genes))
    return mut_dict, primary_site_dict

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    mut_export_df = pd.read_csv(raw_filepath + 'CosmicMutantExportCensus.tsv', delimiter='\t', encoding='ISO-8859-1')

    # FIXME: For testing, remove afterwards
    # mut_export_df = mut_export_df.sample(n=16384)
    hp = hgvs.parser.Parser()
    # load the config
    with open('./config.json', 'r') as f:
        config = json.load(f)
    hgvs_type = config['hgvsType']
    # cols_of_interest = ['Gene name', 'Accession Number', 'ID_sample', 'ID_tumour', 'Primary site', hgvs_type]
    cols_of_interest = config['interestedColumns'] + [hgvs_type]
    logger.info('Doing data cleansing')
    cleansed_df = data_cleansing(mut_export_df, cols_of_interest, hp, hgvs_type)
    # Store the dataframe to ../processed
    cleansed_df.to_csv(processed_filepath + 'CleansedMutantExportCensus.tsv', sep="\t", index=False)
    # Get summary information
    all_tumour_ids, all_genes, all_primary_sites = get_summary_info(cleansed_df)
    # Get data dictionaries
    mut_dict, primary_site_dict = get_data_dictionaries(cleansed_df, hgvs_type, cols_of_interest)
    # Aggregate and store locally
    dataset_dict = {
        'tumour_ids': all_tumour_ids,
        'genes': all_genes,
        'primary_sites': all_primary_sites,
        'mut_dict': mut_dict,
        'primary_site_dict': primary_site_dict
    }
    with open('./data/interim/dataset_dict.pkl', 'wb') as f:
        pickle.dump(dataset_dict, f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()