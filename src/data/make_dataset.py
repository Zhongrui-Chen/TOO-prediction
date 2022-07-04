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
from src.utils.sequences import get_cds_lookup_table, get_flanks, assign_mut_type

# Define the interested columns and primary sites
cols_mut = ['Gene name', 'ID_sample', 'Primary site', 'HGVSC', 'HGVSG']
cols_mut_rename_mapper = {'Gene name': 'gene', 'ID_sample': 'sample_id', 'Primary site': 'site', 'HGVSC': 'hgvsc', 'HGVSG': 'hgvsg'}
cols_cnv = ['gene_name', 'ID_SAMPLE', 'TOTAL_CN', 'MUT_TYPE', 'Chromosome:G_Start..G_Stop']
cols_cnv_rename_mapper = {'gene_name': 'gene', 'ID_SAMPLE': 'sample_id', 'TOTAL_CN': 'total_cn', 'MUT_TYPE': 'mut_type', 'Chromosome:G_Start..G_Stop': 'chrom_range'}
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
    criterion = df['hgvsc'].map(lambda x: is_SNV(x))
    return df.loc[criterion]

def drop_noncoding(df):
    criterion = df['hgvsc'].map(lambda x: is_coding_mut(x))
    return df.loc[criterion]

def preprocess_mut(df):
    # Keep samples of interested primary sites
    l = len(df)
    df = df[df['site'].isin(sites_of_interest)]
    print('  {} rows with non-interested primary sites are removed'.format(l - len(df)))

    # Remove rows with N/A mutation columns
    l = len(df)
    df = df.dropna(how='any', subset=['hgvsc', 'hgvsg'])
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
    df = df.drop_duplicates(subset=['sample_id', 'hgvsc', 'hgvsg'])
    print('  {} duplicate rows are removed'.format(l - len(df)))
    
    return df

def parse_gene(gene):
    if '_' in gene:
        gene = gene.split('_')[0]
    return gene

def make_dataset(df_mut, df_cnv):
    '''
    Make the dataset by constructing the dictionaries
    '''
    loc_of_mut = {} # Location of a column
    for col in df_mut.columns:
        loc_of_mut[col] = df_mut.columns.get_loc(col)
    
    lookup_table = get_cds_lookup_table()

    sample_ids = set()
    cmut_dict = defaultdict(list)
    site_dict = {} # id -> site
    # gmut_dict = {} # (id, chrom) -> [gmut]

    # Check how many rows are invalid
    before = defaultdict(int)
    after = defaultdict(int)
    
    # If a gene was found its sequence mismatches with our FASTA, add it to the blacklist and ignore mutation rows concerning the gene
    gene_blacklist = set()

    for row in tqdm(df_mut.itertuples(index=False), total=len(df_mut)):
        sample_id = row[loc_of_mut['sample_id']]
        before[sample_id] += 1
        gene = row[loc_of_mut['gene']]
        if gene in gene_blacklist:
            continue

        sample_ids.add(sample_id)
        cmut = row[loc_of_mut['hgvsc']]
        gene = row[loc_of_mut['gene']]
        # gmut = row[loc_of['gmut']]
        site = row[loc_of_mut['site']]

        loc, ref, alt = parse_cmut(cmut)
        res = get_flanks(gene, loc, ref, lookup_table)
        if res is not None:
            f5, f3 = res
            # Mappings
            site_dict[sample_id] = site
            mut_type = assign_mut_type(ref, alt, f5, f3)
            cmut_dict[sample_id].append((parse_gene(gene), mut_type))
            after[sample_id] += 1
        else:
            gene_blacklist.add(gene)

    # Check and map the CNVs with the samples
    cnv_dict = defaultdict(list)
    loc_of_cnv = {} # Location of a column
    for col in df_cnv.columns:
        loc_of_cnv[col] = df_cnv.columns.get_loc(col)

    for row in tqdm(df_cnv.itertuples(index=False), total=len(df_cnv)):
        sample_id = row[loc_of_cnv['sample_id']]
        gene = parse_gene(row[loc_of_mut['gene']])
        mut_type = row[loc_of_cnv['mut_type']] # gain or loss
        if sample_id in sample_ids:
            cnv_dict[sample_id].append((gene, mut_type))

    dataset = { # Sample mappings
        'sample_ids': list(sample_ids),
        'cmut_dict': cmut_dict,
        'cnv_dict': cnv_dict,
        # TODO: gmut_dict
        'site_dict': site_dict
    }

    cnt_affected_samples = 0
    cnt_affected_mutations = 0
    for sample_id in dataset['sample_ids']:
        if before[sample_id] > after[sample_id]:
            cnt_affected_samples += 1
            cnt_affected_mutations += before[sample_id] - after[sample_id]
            # print('Before: {}, after: {}'.format(before[sample_id], after[sample_id]))

    print('cnt_affected_samples: {}({}), cnt_affected_mutations: {}({})'.format(cnt_affected_samples, len(dataset['sample_ids']), cnt_affected_mutations, sum(before.values())))

    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='To load the processed TSV file', action='store_true')
    load_flag = parser.parse_args().load
    processed_mut_filepath = './data/processed/ProcessedGenomeScreensMutantExport.tsv'
    raw_mut_filepath = './data/raw/CosmicGenomeScreensMutantExport.tsv'
    raw_cnv_filepath = './data/raw/CosmicCompleteCNA.tsv'

    test_nrows = None # FIXME

    if not load_flag:
        print('Reading the mut file')
        df_mut = pd.read_csv(raw_mut_filepath, sep='\t', encoding='ISO-8859-1', usecols=cols_mut, nrows=test_nrows) # FIXME
        df_mut = df_mut.rename(cols_mut_rename_mapper, axis=1)

        # Preprocess the raw file
        print('Preprocess the mut file')
        df_mut = preprocess_mut(df_mut)

        # Store the processed file
        df_mut.to_csv(processed_mut_filepath, sep="\t", index=False)
        print('The processed Tab-Separated Values file is stored in {}'.format(processed_mut_filepath))
    else:
        df_mut = pd.read_csv(processed_mut_filepath, sep="\t")
        print('Read in the processed mut file')

    print('Reading the CNV file')
    df_cnv = pd.read_csv(raw_cnv_filepath, sep='\t', encoding='ISO-8859-1', usecols=cols_cnv, nrows=test_nrows) # FIXME
    df_cnv = df_cnv.rename(cols_cnv_rename_mapper, axis=1)

    # Make the dataset
    print('Make the dataset')
    dataset = make_dataset(df_mut, df_cnv)

    # Store the dataset
    dataset_filepath = './data/interim/dataset.pkl'
    with open(dataset_filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print('The dataset is stored in {}'.format(dataset_filepath))

if __name__ == '__main__':
    main()