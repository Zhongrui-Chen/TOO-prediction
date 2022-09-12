import pandas as pd
import argparse
# from tqdm import tqdm
# from utils.hgvs import is_coding_mut

cols_mut_rename_mapper = {'ID_sample': 'sample_id', 'Primary site': 'site', 'HGVSG': 'hgvsg'}

def get_columns():
    return cols_mut_rename_mapper.keys()

def preprocess(df_mut):
    # Rename the columns
    # cols_mut_rename_mapper = {'Gene name': 'gene', 'ID_sample': 'sample_id', 'Primary site': 'site', 'HGVSC': 'hgvsc', 'HGVSG': 'hgvsg'}
    df_mut = df_mut.rename(cols_mut_rename_mapper, axis=1)

    # Remove the rows with N/A mutation columns
    lb = len(df_mut)
    df_mut = df_mut.dropna(how='any', subset=['hgvsg'])
    print('  {} rows with N/A mutation columns are removed'.format(lb - len(df_mut)))

    # Remove the rows with invalid HGVSG columns
    lb = len(df_mut)
    df_mut = df_mut.loc[df_mut['hgvsg'].map(lambda hgvs: ':g.' in hgvs)]
    print('  {} rows with invalid HGVSG columns are removed'.format(lb - len(df_mut)))

    # Remove duplicate rows
    lb = len(df_mut)
    df_mut = df_mut.drop_duplicates(subset=['sample_id', 'site', 'hgvsg'])
    print('  {} duplicate rows are removed'.format(lb - len(df_mut)))

    # Leave samples with GWS
    # lb = len(df_mut)
    # gws_sample_ids = set()
    # for row in tqdm(df_mut.itertuples(), total=len(df_mut)):
    #     if not is_coding_mut(row.hgvsc):
    #         gws_sample_ids.add(row.sample_id)
    # df_mut = df_mut.loc[df_mut['sample_id'].map(lambda sample_id: sample_id in gws_sample_ids)]
    # print('{} rows that are not GWS samples are removed'.format(lb - len(df_mut)))

    # df_mut['gene'] = df_mut['gene'].apply(lambda gene: gene.split('_')[0] if '_' in gene else gene)

    return df_mut

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='The name of the preprocessed version')
    ver_name = parser.parse_args().name

    raw_mut_filepath = './data/raw/CosmicGenomeScreensMutantExport.tsv'
    preprocessed_mut_filepath = './data/processed/CosmicGenomeScreensMutantExport_{}.tsv'.format(ver_name)
    
    print('Reading the mutation file')
    df_mut = pd.read_csv(raw_mut_filepath, sep='\t', usecols=get_columns())

    # Preprocess the dataframe
    df_mut = preprocess(df_mut)

    # Save the processed file
    df_mut.to_csv(preprocessed_mut_filepath, sep='\t', index=False)

    # Sample 10 rows to display
    print(df_mut.sample(n=10))

if __name__ == '__main__':
    main()