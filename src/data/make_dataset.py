# -----------------------------------------------------------------
# Make the dataset by processing the COSMIC mutation TSV file
# Steps:
#   1. Filter and cleanse the dataframe
#   2. Make the dataset by constructing an aggregative dictionary
#   3. Pickle the dataset dict and store it in the disk
# -----------------------------------------------------------------

import pickle
from tqdm import tqdm
import pandas as pd
import hgvs.parser
from src.utils.hgvs_parsing import is_hgvs_valid, parse_chrom

# Define the interested columns and primary sites
cols_of_interest = ['Gene name', 'ID_sample', 'Primary site', 'Tier', 'HGVSC', 'HGVSG']
sites_of_interest = [
    'kidney', 'skin', 'liver', 'breast', 'ovary', 'haematopoietic_and_lymphoid_tissue',
    'prostate', 'pancreas', 'central_nervous_system', 'lung', 'oesophagus', 'thyroid', 'bone' ]

# Instantiate the hgvs parser
hp = hgvs.parser.Parser()

def data_filtering(df):
    '''
    Filter the rows as per predefined standards
    '''
    # Select interested columns
    df = df[cols_of_interest]

    # Filter by gene tiers
    len_before = len(df)
    df = df[df['Tier'] == 1]
    print('{} rows with tier 2 genes were removed'.format(len_before - len(df)))

    # Filter by primary sites
    len_before = len(df)
    df = df[df['Primary site'].isin(sites_of_interest)]
    print('{} rows with non-interested primary sites were removed'.format(len_before - len(df)))
    
    return df

def data_cleansing(df):
    '''
    Cleanse the dataset by removing unhelpful rows
    '''
    # Remove duplicates
    len_before = len(df)
    df = df.drop_duplicates(subset=['ID_sample', 'HGVSC'])
    print('{} duplicate rows are removed'.format(len_before - len(df)))

    # Remove rows with N/A variant information
    NaNs = df['HGVSC'].isna()
    idcs = [idx for idx in range(len(df)) if not NaNs.iloc[idx]]
    print('{} rows with N/A variant information are removed'.format(len(df) - len(idcs)))
    df = df.iloc[idcs, :]

    # Remove rows with non-coding variants
    print('Removing rows with non-coding variants')
    len_before = len(idcs)
    idcs = []
    for idx, hgvsc in enumerate(tqdm(df['HGVSC'])):
        var = hp.parse_hgvs_variant(hgvsc)
        if is_hgvs_valid(var):
            idcs.append(idx)
    print('{} rows with non-coding variants are removed'.format(len_before - len(idcs)))
    df = df.iloc[idcs, :]

    return df

def get_summary_info(df):
    all_sample_ids = list(set(df['ID_sample']))
    all_genes = list(set(df['Gene name']))
    return all_sample_ids, all_genes

def make_dataset(df, all_sample_ids, all_genes):
    '''
    Make the dataset by constructing the dictionaries
    '''
    loc_of = {} # Location of a column
    for col in cols_of_interest:
        loc_of[col] = df.columns.get_loc(col)
    cmut_dict = {} # (sample_id, gene) -> [mutations]
    gmut_dict = {} # (sample_id, chromosome) -> [mutations]
    primary_site_dict = {} # sample_id -> primary_site
    for row in df.itertuples(index=False):
        sample_id = row[loc_of['ID_sample']]
        gene = row[loc_of['Gene name']]
        cmut = row[loc_of['HGVSC']]
        gmut = 'chr' + row[loc_of['HGVSG']] # Add a prefix to make the parser work
        gvar = hp.parse_hgvs_variant(gmut)
        primary_site = row[loc_of['Primary site']]
        # Add primary site
        if sample_id not in primary_site_dict:
            primary_site_dict[sample_id] = primary_site
        # Add coding mutations
        if (sample_id, gene) not in cmut_dict:
            cmut_dict[(sample_id, gene)] = [cmut]
        else:
            cmut_dict[(sample_id, gene)].append(cmut)
        # Add genomic mutations
        chrom = parse_chrom(gvar)
        if (sample_id, chrom) not in gmut_dict:
            gmut_dict[(sample_id, chrom)] = [gmut]
        else:
            gmut_dict[(sample_id, chrom)].append(gmut)
    dataset = {
        # Summary info
        'sample_ids': all_sample_ids,
        'genes': all_genes,
        'primary_sites': sites_of_interest,
        # Sample mappings
        'cmut_dict': cmut_dict,
        'gmut_dict': gmut_dict,
        'primary_site_dict': primary_site_dict
    }
    return dataset

def main():
    # Read in the COSMIC data as dataframe
    df = pd.read_csv('./data/raw/CosmicMutantExportCensus.tsv', delimiter='\t', encoding='ISO-8859-1')
    # df = df.sample(n=16384) # FIXME

    # Filter and cleanse the raw dataframe
    print('[Data filtering]')
    df = data_filtering(df)
    print('[Data cleansing]')
    df = data_cleansing(df)

    # Get the summary info
    print('There are {} primary sites in total'.format(len(sites_of_interest)))
    all_sample_ids, all_genes = get_summary_info(df)
    print('There are {} samples in total'.format(len(all_sample_ids)))
    print('There are {} genes in total'.format(len(all_genes)))

    # Make the dataset
    print('[Making the dataset]')
    dataset = make_dataset(df, all_sample_ids, all_genes)
    # print(dataset['gmut_dict']) # FIXME
    filepath = './data/interim/dataset.pkl' # FIXME
    print('The dataset is stored in {}'.format(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    main()