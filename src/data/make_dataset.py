# -----------------------------------------------------------------
# Make the dataset by processing the COSMIC mutation TSV file
# Steps:
#   1. Filter and cleanse the dataframe
#   2. Make the dataset by constructing an aggregative dictionary
#   3. Pickle the dataset dict and store it in the disk
# -----------------------------------------------------------------

import pickle
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import argparse
# import hgvs.parser
from src.utils.hgvs_parsing import is_SNV, is_coding_mut, parse_cmut
from src.utils.sequences import get_cds_lookup_table, get_flanks, InvalidMutError

# Define the interested columns and primary sites
cols_of_interest = ['Gene name', 'ID_sample', 'Primary site', 'Genome-wide screen', 'Sample Type', 'HGVSC', 'HGVSG']
# sites_of_interest = [
#     'kidney', 'skin', 'liver', 'breast', 'ovary', 'haematopoietic_and_lymphoid_tissue',
#     'prostate', 'pancreas', 'central_nervous_system', 'lung', 'oesophagus', 'thyroid', 'bone' ]
sites_of_interest = [
    'breast', 'kidney', 'liver', 'ovary', 'prostate',
    'endometrium', 'large_intestine', 'lung', 'pancreas', 'skin'
] # The same primary sites that are used for classifiers in [Marquard et al.](https://bmcmedgenomics.biomedcentral.com/articles/10.1186/s12920-015-0130-0)

# Instantiate the hgvs parser // we no longer use it limited by the overhead
# hp = hgvs.parser.Parser()

def keep_SNVs(df):
    criterion = df['HGVSC'].map(lambda x: is_SNV(x))
    return df.loc[criterion]

def drop_noncoding(df):
    criterion = df['HGVSC'].map(lambda x: is_coding_mut(x))
    return df.loc[criterion]

def preprocess(df):
    # Keep samples of interested primary sites
    l = len(df)
    df = df[df['Primary site'].isin(sites_of_interest)]
    print('  {} rows with non-interested primary sites are removed'.format(l - len(df)))

    # Remove rows with N/A mutation columns
    l = len(df)
    df = df.dropna(how='any', subset=['HGVSC', 'HGVSG'])
    print('  {} rows with N/A mutation columns are removed'.format(l - len(df)))

    # Currently, only keep the SNVs
    l = len(df)
    df = keep_SNVs(df)
    print('  {} rows which are not SNVs are removed'.format(l - len(df)))

    # Remove rows with non-coding variants
    l = len(df)
    df = drop_noncoding(df)
    print('  {} rows with non-coding variants are removed'.format(l - len(df)))

    # Remove duplicates
    l = len(df)
    df = df.drop_duplicates(subset=['ID_sample', 'HGVSC', 'HGVSG'])
    print('  {} duplicate rows are removed'.format(l - len(df)))
    
    return df

def assign_mut_type(ref, alt, f5, f3):
    mapping = {
        'G': 'C',
        'C': 'G',
        'A': 'T',
        'T': 'A'
    }
    if ref == 'G' or ref == 'A':
        return (f5, (mapping[ref], mapping[alt]), f3)
    else:
        return (f5, (ref, alt), f3)

def make_dataset(df):
    '''
    Make the dataset by constructing the dictionaries
    '''
    loc_of = {} # Location of a column
    for col in cols_of_interest:
        loc_of[col] = df.columns.get_loc(col)
    lookup_table = get_cds_lookup_table()

    sample_ids = set()
    cmut_dict = defaultdict(list)
    site_dict = {} # id -> site
    # gmut_dict = {} # (id, chrom) -> [gmut]
    before = defaultdict(int)
    after = defaultdict(int)

    gene_blacklist = set()

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        sample_id = row[loc_of['ID_sample']]
        
        before[sample_id] += 1

        gene = row[loc_of['Gene name']]
        if gene in gene_blacklist:
            continue

        sample_ids.add(sample_id)

        cmut = row[loc_of['HGVSC']]
        # gmut = row[loc_of['gmut']]
        site = row[loc_of['Primary site']]

        loc, ref, alt = parse_cmut(cmut)
        res = get_flanks(gene, loc, ref, lookup_table)
        if res is not None:
            f5, f3 = res
            # Mappings
            site_dict[sample_id] = site
            mut_type = assign_mut_type(ref, alt, f5, f3)
            cmut_dict[sample_id].append(mut_type)        
            after[sample_id] += 1
        else:
            gene_blacklist.add(gene)

    dataset = { # Sample mappings
        'sample_ids': list(sample_ids),
        'cmut_dict': cmut_dict,
        # TODO: gmut_dict
        'site_dict': site_dict
    }

    cnt_affected_samples = 0
    cnt_affected_mutations = 0
    for sample_id in dataset['sample_ids']:
        if before[sample_id] > after[sample_id]:
            cnt_affected_samples += 1
            cnt_affected_mutations += before[sample_id] - after[sample_id]
            print('Before: {}, after: {}'.format(before[sample_id], after[sample_id]))

    print('cnt1: {}({}), cnt2: {}({})'.format(cnt_affected_samples, len(dataset['sample_ids']), cnt_affected_mutations, sum(before.values())))

    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='To load the processed TSV file', action='store_true')
    load_flag = parser.parse_args().load
    processed_tsv_filepath = './data/processed/ProcessedGenomeScreensMutantExport.tsv'

    if not load_flag:
        # Read in the raw file
        print('Read in the raw file')
        raw_tsv_filepath = './data/raw/CosmicGenomeScreensMutantExport.tsv'
        df = pd.read_csv(raw_tsv_filepath, sep='\t', encoding='ISO-8859-1', usecols=cols_of_interest)

        # Preprocess the raw file
        print('Preprocess the raw file')
        df = preprocess(df)

        # Store the processed file
        df.to_csv(processed_tsv_filepath, sep="\t", index=False)
        print('The processed Tab-Separated Values file is stored in {}'.format(processed_tsv_filepath))
    else:
        df = pd.read_csv(processed_tsv_filepath, sep="\t")
        print('Read in the processed file')

    # Make the dataset
    print('Make the dataset')
    dataset = make_dataset(df) # FIXME

    # Store the dataset
    dataset_filepath = './data/interim/dataset.pkl'
    with open(dataset_filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print('The dataset is stored in {}'.format(dataset_filepath))

if __name__ == '__main__':
    main()