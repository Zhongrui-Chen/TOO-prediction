import pandas as pd
from tqdm import tqdm
from src.utils.hgvs_parsing import is_coding_mut

cols_mut_rename_mapper = {'Gene name': 'gene', 'ID_sample': 'sample_id', 'Primary site': 'site', 'HGVSC': 'hgvsc', 'HGVSG': 'hgvsg'}
sites_of_interest = [
'breast', 'kidney', 'liver', 'ovary', 'prostate', 'endometrium', 'large_intestine', 'lung', 'pancreas', 'skin' # 10 classes
# 'ovary', 'lung', 'large_intestine', 'kidney', 'endometrium', 'breast' # 6 classes
] # The same primary sites that are used for classifiers in [Marquard et al.](https://bmcmedgenomics.biomedcentral.com/articles/10.1186/s12920-015-0130-0)

def preprocess(df_mut):
    # Rename the columns
    df_mut = df_mut.rename(cols_mut_rename_mapper, axis=1)

    # Select samples of interested primary sites
    lb = len(df_mut)
    df_mut = df_mut[df_mut['site'].isin(sites_of_interest)]
    print('  {} rows with non-interested primary sites are removed'.format(lb - len(df_mut)))

    # Remove rows with N/A mutation columns
    lb = len(df_mut)
    # df_mut = df_mut.dropna(how='any', subset=['cmut', 'gmut', 'pmut'])
    df_mut = df_mut.dropna(how='any', subset=['hgvsc', 'hgvsg'])
    print('  {} rows with N/A mutation columns are removed'.format(lb - len(df_mut)))

    # Remove duplicate rows
    lb = len(df_mut)
    # df_mut = df_mut.drop_duplicates(subset=['sample_id', 'cmut', 'gmut', 'pmut'])
    df_mut = df_mut.drop_duplicates(subset=['sample_id', 'site', 'gene', 'hgvsc', 'hgvsg'])
    print('  {} duplicate rows are removed'.format(lb - len(df_mut)))

    # Leave samples with GWS
    lb = len(df_mut)
    gws_sample_ids = set()
    for row in tqdm(df_mut.itertuples(), total=len(df_mut)):
        if not is_coding_mut(row.hgvsc):
            gws_sample_ids.add(row.sample_id)
    df_mut = df_mut.loc[df_mut['sample_id'].map(lambda sample_id: sample_id in gws_sample_ids)]
    print('{} rows that are not GWS samples are removed'.format(lb - len(df_mut)))

    def revise_gene(gene):
        if '_' in gene:
            gene = gene.split('_')[0]
        return gene

    df_mut['gene'] = df_mut['gene'].apply(revise_gene)

    return df_mut

def main():
    raw_mut_filepath = './data/raw/CosmicGenomeScreensMutantExport.tsv'
    preprocessed_mut_filepath = './data/processed/CosmicGenomeScreensMutantExport.tsv'
    cols_mut = ['Gene name', 'ID_sample', 'Primary site', 'HGVSC', 'HGVSG']
    print('Reading the mutation file')
    df_mut = pd.read_csv(raw_mut_filepath, sep='\t', encoding='ISO-8859-1', usecols=cols_mut)
    df_mut = preprocess(df_mut)
    df_mut.to_csv(preprocessed_mut_filepath, sep='\t', index=False)
    print(df_mut.sample(n=10))

if __name__ == '__main__':
    main()