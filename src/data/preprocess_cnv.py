import pandas as pd
from matplotlib import pyplot as plt

def main():
    raw_cnv_filepath = './data/raw/CosmicCompleteCNA.tsv'
    processed_cnv_filepath = './data/processed/CosmicCompleteCNA.tsv'
    cnv_cols_mapper = {
        'CNV_ID': 'cnv_id', 'ID_SAMPLE': 'sample_id', 'MINOR_ALLELE': 'minor',
        'TOTAL_CN': 'total_cn', 'MUT_TYPE': 'gain_loss', 'Chromosome:G_Start..G_Stop': 'genomic_coordinates'
    }

    print('Reading the file...')
    df_cnv = pd.read_csv(raw_cnv_filepath, sep='\t', usecols=cnv_cols_mapper.keys())
    
    print('Preprocess...')
    df_cnv = df_cnv.rename(cnv_cols_mapper, axis=1)
    # Remove duplicates
    df_cnv = df_cnv.drop_duplicates(subset=['cnv_id'])
    lb = len(df_cnv)
    # Remove CNVs with negative Copy Numbers
    df_cnv = df_cnv.loc[df_cnv['total_cn'].map(lambda x: x >= 0)]
    # Remove CNVs with inavailable CN and genomic coordinates
    # df_cnv = df_cnv.dropna(subset=['total_cn', 'genomic_coordinates', 'minor', 'gain_loss'])
    df_cnv = df_cnv.dropna(subset=['total_cn', 'genomic_coordinates'])
    print('{} rows are removed, out of {} (Remain percentage {:.2%})'.format(lb - len(df_cnv), lb, len(df_cnv) / lb))
    # df_cnv[['total_cn', 'minor']] = df_cnv[['total_cn', 'minor']].astype(int)
    df_cnv[['total_cn']] = df_cnv[['total_cn']].astype(int)

    # Record the sample IDs
    cnv_sample_id_list_filepath = './data/processed/CNVSampleIDList.tsv'
    sample_ids = df_cnv['sample_id'].drop_duplicates()
    sample_ids.to_csv(cnv_sample_id_list_filepath, index=False, sep='\t')

    # Display head and save the processed file
    print(df_cnv.head())
    df_cnv.to_csv(processed_cnv_filepath, sep='\t')

if __name__ == "__main__":
    main()