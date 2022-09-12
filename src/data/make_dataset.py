from tqdm import tqdm
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='The name of the preprocessed version')
    parser.add_argument('--test', help='Test with a number of rows', type=int)
    parser.add_argument('--site', type=bool)
    args = parser.parse_args()
    site_flag = False if not args.site else True
    ver_name = args.name

    preprocessed_mut_filepath = './data/processed/CosmicGenomeScreensMutantExport_{}.tsv'.format(ver_name)
    preprocessed_cnv_filepath = './data/processed/CosmicCompleteCNA.tsv'
    dataset_folder = './data/interim/dataset'
    sample_mut_filepath = dataset_folder + '/samples_mut'
    sample_cnv_filepath = dataset_folder + '/samples_cnv'
    site_filepath = dataset_folder + '/sites.tsv'

    print('Reading the preprocessed files...')
    df_mut = pd.read_csv(preprocessed_mut_filepath, sep='\t', nrows=args.test)
    df_cnv = pd.read_csv(preprocessed_cnv_filepath, sep='\t', nrows=args.test)

    # If needed, generate the sites file
    if site_flag:
        df_site = df_mut[['sample_id', 'site']].drop_duplicates()
        df_site = df_site.set_index('sample_id')  
        df_site.to_csv(site_filepath, sep='\t')
    
    # Generate a mutation list and a CNV list for each sample
    for sample_id in tqdm(set(df_mut['sample_id'])):
    # for sample_id in tqdm(set(df_cnv['sample_id'])):
        # Aggregate all mutations and CNVs with the sample
        df_mut_sample = df_mut[df_mut['sample_id'] == sample_id]
        df_cnv_sample = df_cnv[df_cnv['sample_id'] == sample_id]

        # Save the sample file
        sample_filename = '/' + str(sample_id) + '.tsv'
        df_mut_sample['hgvsg'].to_csv(sample_mut_filepath + sample_filename, sep='\t', index=False)
        df_cnv_sample[['total_cn', 'genomic_coordinates', 'gain_loss', 'minor']].to_csv(sample_cnv_filepath + sample_filename, sep='\t', index=False)

if __name__ == '__main__':
    main()