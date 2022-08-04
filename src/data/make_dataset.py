from tqdm import tqdm
import pandas as pd
import argparse

def revise_gene(gene):
    if '_' in gene:
        gene = gene.split('_')[0]
    return gene

def make_dataset(df_mut):
    sample_ids = set(df_mut['sample_id'])
    df_dict = {}
    site_dict = {}
    for sample_id in sample_ids:
        df_sample = df_mut[df_mut['sample_id'] == sample_id][['gene', 'hgvsg']]
        df_dict[sample_id] = df_sample

    return df_dict, site_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Test with only a number of rows', type=int)
    parser.add_argument('--skip', nargs='+')
    args = parser.parse_args()
    test_nrows = args.test
    skip_files = args.skip

    preprocessed_mut_filepath = './data/processed/CosmicGenomeScreensMutantExport.tsv'
    dataset_folder = './data/interim/dataset'
    sample_filepath = dataset_folder + '/samples'
    site_filepath = dataset_folder + '/sites.tsv'

    print('Reading the preprocessed mutation file')
    if not skip_files or 'mut' not in skip_files:
        df_mut = pd.read_csv(preprocessed_mut_filepath, sep='\t', nrows=test_nrows)
        sample_ids = list(set(df_mut['sample_id']))
        site_list = []
        for sample_id in tqdm(sample_ids):
            df_sample = df_mut[df_mut['sample_id'] == sample_id]
            file_name = '/' + str(sample_id) + '.tsv'
            df_sample[['gene', 'hgvsc', 'hgvsg']].to_csv(sample_filepath + file_name, sep='\t', index=False)
            site_list.append(df_sample['site'].iloc[0])
        df_site = pd.DataFrame.from_dict({'sample_id': sample_ids, 'site': site_list})
        df_site.to_csv(site_filepath, sep='\t', index=False)

if __name__ == '__main__':
    main()