import pickle
from sklearn import datasets
from tqdm import tqdm
import pandas as pd
import hgvs.parser
from src.utils.hgvs_parsing import is_hgvs_valid

# Define the interested columns and primary sites
cols_of_interest = ['Gene name', 'ID_sample', 'Primary site', 'Tier', 'HGVSC', 'HGVSG']
sites_of_interest = [
    'kidney', 'skin', 'liver', 'breast', 'ovary', 'haematopoietic_and_lymphoid_tissue',
    'prostate', 'pancreas', 'central_nervous_system', 'lung', 'oesophagus', 'thyroid', 'bone' ]

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
    # Instantiate the hgvs parser
    hp = hgvs.parser.Parser()

    # Remove duplicates
    # len_before_removing_duplicates = len(df)
    # df = df.drop_duplicates(subset=['ID_sample', hgvs_type])
    # print('{} duplicate rows were removed'.format(len_before_removing_duplicates - len(df)))

    # Get indices of rows with N/A variant information
    NaNs = df['HGVSC'].isna()
    idcs = [idx for idx in range(len(df)) if not NaNs.iloc[idx]]
    print('{} rows with N/A variant information will be removed'.format(len(df) - len(idcs)))

    # Get indices of rows with non-coding variants
    print('Removing rows with non-coding variants')
    len_before = len(idcs)
    for idx in tqdm(idcs):
        if not is_hgvs_valid(str(df['HGVSC'].iloc[idx]), hp):
            idcs.remove(idx)
    print('{} rows with non-coding variants will be removed'.format(len_before - len(idcs)))

    # Remove the rows
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
    mut_dict = {} # (sample_id, gene) -> [mutations]
    primary_site_dict = {} # sample_id -> primary_site
    for row in df.itertuples(index=False):
        sample_id = row[loc_of['ID_sample']]
        gene = row[loc_of['Gene name']]
        mut = row[loc_of['HGVSC']]
        primary_site = row[loc_of['Primary site']]
        if sample_id not in primary_site_dict:
            primary_site_dict[sample_id] = primary_site
        if (sample_id, gene) not in mut_dict:
            mut_dict[(sample_id, gene)] = [mut]
        else:
            mut_dict[(sample_id, gene)].append(mut)
    dataset = {
        # Summary info
        'sample_ids': all_sample_ids,
        'genes': all_genes,
        'primary_sites': sites_of_interest,
        # Sample mappings
        'mut_dict': mut_dict,
        'primary_site_dict': primary_site_dict
    }
    return dataset

def main():
    # Read in the COSMIC data as dataframe
    df = pd.read_csv('./data/raw/CosmicMutantExportCensus.tsv', delimiter='\t', encoding='ISO-8859-1')
    # Filter and cleanse the raw dataframe
    print('Data filtering...')
    df = data_filtering(df)
    print('Data cleansing...')
    df = data_cleansing(df)
    # Get the summary info
    print('There are {} primary sites in total'.format(len(sites_of_interest)))
    all_sample_ids, all_genes = get_summary_info(df)
    print('There are {} samples in total'.format(len(all_sample_ids)))
    print('There are {} genes in total'.format(len(all_genes)))
    # Make the dataset
    dataset = make_dataset(df, all_sample_ids, all_genes)
    with open('./data/interim/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    main()