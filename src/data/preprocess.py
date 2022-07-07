from tqdm import tqdm
import pandas as pd
import argparse
# import hgvs.parser
from src.utils.hgvs_parsing import is_SNV, is_coding_mut, parse_gmut, parse_snv, parse_cmut, parse_mut_type, parse_indel
from src.utils.sequences import get_cds_lookup_table, get_flanks, assign_mut_type, is_matched_seq

# Define the interested columns and primary sites
cols_mut = ['Gene name', 'ID_sample', 'Primary site', 'Mutation Description', 'Mutation AA', 'HGVSC', 'HGVSG']
cols_mut_rename_mapper = {'Gene name': 'gene', 'ID_sample': 'sample_id', 'Primary site': 'site', 'Mutation Description': 'mut_effect', 'Mutation AA': 'pmut', 'HGVSC': 'cmut', 'HGVSG': 'gmut'}
cols_cnv = ['gene_name', 'ID_SAMPLE', 'TOTAL_CN', 'MUT_TYPE', 'Chromosome:G_Start..G_Stop']
cols_cnv_rename_mapper = {'gene_name': 'gene', 'ID_SAMPLE': 'sample_id', 'TOTAL_CN': 'total_cn', 'MUT_TYPE': 'mut_type', 'Chromosome:G_Start..G_Stop': 'chrom_range'}
# cols_ge = ['SAMPLE_ID', 'GENE_NAME', 'Z_SCORE']
# cols_ge_rename_mapper = {'SAMPLE_ID': 'sample_id', 'GENE_NAME': 'gene', 'Z_SCORE': 'z_score'}
# sites_of_interest = [
#     'kidney', 'skin', 'liver', 'breast', 'ovary', 'haematopoietic_and_lymphoid_tissue',
#     'prostate', 'pancreas', 'central_nervous_system', 'lung', 'oesophagus', 'thyroid', 'bone' ]
sites_of_interest = [
    'breast', 'kidney', 'liver', 'ovary', 'prostate', 'endometrium', 'large_intestine', 'lung', 'pancreas', 'skin' # 10 classes
    # 'ovary', 'lung', 'large_intestine', 'kidney', 'endometrium', 'breast' # 6 classes
] # The same primary sites that are used for classifiers in [Marquard et al.](https://bmcmedgenomics.biomedcentral.com/articles/10.1186/s12920-015-0130-0)

cds_fasta_filepath = './data/external/All_COSMIC_Genes.fasta'

# def keep_SNVs(df):
#     criterion = df['cmut'].map(lambda x: is_SNV(x))
#     return df.loc[criterion]

def drop_noncoding(df):
    criterion = df['cmut'].map(lambda x: is_coding_mut(x))
    return df.loc[criterion]

def drop_blacklisted_genes(df, gene_blacklist):
    criterion = df['gene'].map(lambda x: x not in gene_blacklist)
    return df.loc[criterion]

def preprocess_mut(df_mut):
    # Renaming
    df_mut = df_mut.rename(cols_mut_rename_mapper, axis=1)

    # Select samples of interested primary sites
    lb = len(df_mut)
    df_mut = df_mut[df_mut['site'].isin(sites_of_interest)]
    print('  {} rows with non-interested primary sites are removed'.format(lb - len(df_mut)))

    # Remove rows with N/A mutation columns
    lb = len(df_mut)
    df_mut = df_mut.dropna(how='any', subset=['cmut', 'gmut', 'pmut'])
    print('  {} rows with N/A mutation columns are removed'.format(lb - len(df_mut)))

    # # Remove rows with non-coding variants
    # l = len(df_mut)
    # df_mut = drop_noncoding(df_mut)
    # print('  {} rows with non-coding variants are removed'.format(l - len(df_mut)))

    # Remove duplicate rows
    lb = len(df_mut)
    df_mut = df_mut.drop_duplicates(subset=['sample_id', 'cmut', 'gmut', 'pmut'])
    print('  {} duplicate rows are removed'.format(lb - len(df_mut)))

    # Get 5' and 3' flanks of each mutation, meanwhile exclude the mutations on genes which we have not the valid sequences of
    df_mut = pd.concat([df_mut, pd.DataFrame(columns=['coding', 'mut_type'], dtype=str)], axis=1)
    gene_blacklist = set() # If a gene was found its sequence mismatches with our FASTA, add it to the blacklist and ignore mutation rows concerning the gene
    lookup_table = get_cds_lookup_table(cds_fasta_filepath)
    for row in tqdm(df_mut.itertuples(), total=len(df_mut)):
        row_idx = row.Index
        gene = row.gene

        # Parse the coding mutation
        cpos, cmut = parse_cmut(row.cmut)
        # Parse the mutational type
        mut_type = parse_mut_type(cmut)
        df_mut.at[row_idx, 'mut_type'] = mut_type
        if is_coding_mut(row.cmut) and gene not in gene_blacklist:
            df_mut.at[row_idx, 'coding'] = 1
            # For SNVs, parse the ref and alt nucleotides
            if mut_type == 'SNV':
                ref, alt = parse_snv(cmut)
                if is_matched_seq(gene, cpos, ref, lookup_table):
                    f5, f3 = get_flanks(gene, cpos, lookup_table)
                    df_mut.at[row_idx, 'cmut'] = (f5, (ref, alt), f3)
                else:
                    gene_blacklist.add(gene)
            # For INDELs, parse the alt sequence
            elif mut_type == 'INDEL':
                altseq = parse_indel(cmut)
                df_mut.at[row_idx, 'cmut'] = (cpos, altseq)
            else:
                df_mut.at[row_idx, 'cmut'] = cpos
        else:
            df_mut.at[row_idx, 'coding'] = 0
            df_mut.at[row_idx, 'cmut'] = 'NaN'
        # Parse the genomic mutation
        chrom, gpos = parse_gmut(row.gmut)
        df_mut.at[row_idx, 'gmut'] = ((chrom, gpos))
        # Revise the gene name by removing the accession number
        df_mut.at[row_idx, 'gene'] = revise_gene(gene)
        # Compact the mutational effect string
        df_mut.at[row_idx, 'mut_effect'] = row.mut_effect.replace(' ', '').replace('-', '')

    # df_mut = drop_blacklisted_genes(df_mut, gene_blacklist)
    return df_mut

def preprocess_cnv(df_mut, df_cnv):
    df_cnv = df_cnv.rename(cols_cnv_rename_mapper, axis=1)
    l = len(df_cnv)
    criterion = df_cnv['sample_id'].map(lambda x: x in df_mut['sample_id'])
    df_cnv = df_cnv.loc[criterion]
    print('{} rows in the CNV file are remained out of {}'.format(len(df_cnv), l))
    return df_cnv

def revise_gene(gene):
    if '_' in gene:
        gene = gene.split('_')[0]
    return gene

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Test with a number of rows', type=int)
    n_testrows = parser.parse_args().test

    raw_mut_filepath = './data/raw/CosmicGenomeScreensMutantExport.tsv'
    processed_mut_filepath = './data/processed/ProcessedGenomeScreensMutantExport.tsv'
    raw_cnv_filepath = './data/raw/CosmicCompleteCNA.tsv'
    processed_cnv_filepath = './data/processed/ProcessedCompleteCNA.tsv'

    print('Reading the mutation file')
    df_mut = pd.read_csv(raw_mut_filepath, sep='\t', encoding='ISO-8859-1', usecols=cols_mut, nrows=n_testrows) # FIXME

    # Preprocess the raw mutation file
    print('Preprocess the mutation file')
    df_mut = preprocess_mut(df_mut)

    print(df_mut[df_mut.coding == 1].head())

    # Store the processed mut file
    if not n_testrows:
        df_mut.to_csv(processed_mut_filepath, sep="\t", index=False)
        print('The processed mutation file is stored in {}'.format(processed_mut_filepath))
    
    print('Reading the CNV file')
    df_cnv = pd.read_csv(raw_cnv_filepath, sep='\t', encoding='ISO-8859-1', usecols=cols_cnv, nrows=n_testrows)

    # Preprocess the raw CNV file
    print('Preprocess the CNV file')
    df_cnv = preprocess_cnv(df_mut, df_cnv)

    # Store the processed CNV file
    if not n_testrows:
        df_cnv.to_csv(processed_cnv_filepath, sep="\t", index=False)
        print('The processed CNV file is stored in {}'.format(processed_cnv_filepath))

    # print(df_mut.head(10))

if __name__ == '__main__':
    main()