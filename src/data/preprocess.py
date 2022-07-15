from collections import defaultdict
from math import floor, ceil, inf
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
# import hgvs.parser
from src.utils.hgvs_parsing import is_coding_mut, parse_hgvs, parse_snv_edit
from src.utils.sequences import get_chrom_by_idx, get_chrom_idx_by_chrom, get_flanks, get_genomic_sequences, nuc_at, sanity_check

# Define the interested columns and primary sites
# cols_mut = ['Gene name', 'ID_sample', 'Primary site', 'Mutation Description', 'Mutation AA', 'HGVSC', 'HGVSG']
cols_mut = ['Gene name', 'ID_sample', 'Primary site', 'Mutation Description', 'HGVSC', 'HGVSG']
# cols_mut_rename_mapper = {'Gene name': 'gene', 'ID_sample': 'sample_id', 'MUTATION_ID': 'mut_id', 'Primary site': 'site', 'Mutation Description': 'mut_effect', 'Mutation AA': 'pmut', 'HGVSC': 'cmut', 'HGVSG': 'gmut'}
# cols_mut_rename_mapper = {'Gene name': 'gene', 'ID_sample': 'sample_id', 'MUTATION_ID': 'mut_id', 'Primary site': 'site', 'Mutation Description': 'mut_effect', 'HGVSC': 'cmut', 'HGVSG': 'gmut'}
cols_mut_rename_mapper = {'Gene name': 'gene', 'ID_sample': 'sample_id', 'Primary site': 'site', 'Mutation Description': 'mut_effect', 'HGVSC': 'hgvsc', 'HGVSG': 'hgvsg'}
cols_cnv = ['gene_name', 'ID_SAMPLE', 'TOTAL_CN', 'MUT_TYPE']
cols_cnv_rename_mapper = {'gene_name': 'gene', 'ID_SAMPLE': 'sample_id', 'TOTAL_CN': 'total_cn', 'MUT_TYPE': 'cnv_mut_type'}
# cols_ge = ['SAMPLE_ID', 'GENE_NAME', 'Z_SCORE']
# cols_ge_rename_mapper = {'SAMPLE_ID': 'sample_id', 'GENE_NAME': 'gene', 'Z_SCORE': 'z_score'}
# sites_of_interest = [
#     'kidney', 'skin', 'liver', 'breast', 'ovary', 'haematopoietic_and_lymphoid_tissue',
#     'prostate', 'pancreas', 'central_nervous_system', 'lung', 'oesophagus', 'thyroid', 'bone' ]
sites_of_interest = [
    'breast', 'kidney', 'liver', 'ovary', 'prostate', 'endometrium', 'large_intestine', 'lung', 'pancreas', 'skin' # 10 classes
    # 'ovary', 'lung', 'large_intestine', 'kidney', 'endometrium', 'breast' # 6 classes
] # The same primary sites that are used for classifiers in [Marquard et al.](https://bmcmedgenomics.biomedcentral.com/articles/10.1186/s12920-015-0130-0)

# cds_fasta_filepath = './data/external/All_COSMIC_Genes.fasta'
# genomic_fasta_filepath = './data/external/GCF_000001405.40_GRCh38.p14_genomic.fna'
genomic_fasta_filepath = './data/external/GCF_000001405.26_GRCh38_genomic.fna'
num_chromosomes = 24

# def keep_SNVs(df):
#     criterion = df['cmut'].map(lambda x: is_SNV(x))
#     return df.loc[criterion]

# def drop_noncoding(df):
#     criterion = df['cmut'].map(lambda x: is_coding_mut(x))
#     return df.loc[criterion]

# def drop_blacklisted_genes(df, gene_blacklist):
#     criterion = df['gene'].map(lambda x: x not in gene_blacklist)
#     return df.loc[criterion]

def preprocess_mut(df_mut: pd.DataFrame) -> pd.DataFrame:
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

    # Get the genomic sequences
    genomic_seqs = get_genomic_sequences(genomic_fasta_filepath)
    # Calculate the numbers of bins in each chromosome
    num_bins = np.zeros(num_chromosomes, dtype=int)
    bin_len = 1000000
    for chrom_idx in range(1, num_chromosomes+1):
        chrom = get_chrom_by_idx(chrom_idx)
        num_bins[chrom_idx-1] = ceil(len(genomic_seqs[chrom]) / bin_len)
    print(np.cumsum(num_bins))
    
    # The processed results
    mut_dict = defaultdict(list)
    gws_sample_ids = set() # Sample IDs that went through genome-wide sequencing
    # Statistical evaluation
    valid_sample_ids = set()
    invalid_sample_ids = set()
    num_valid_rows = 0
    num_invalid_rows = 0
    num_snvs = defaultdict(int)

    for row in tqdm(df_mut.itertuples(), total=len(df_mut)):
        # Parse the HGVSG
        chrom, ref_type, pos, edit, mut_type = parse_hgvs(row.hgvsg)
        if ref_type != 'g':
            continue
        # If we find a noncoding genomic variant, it indicates the associated sample has gone through the GWS
        if not is_coding_mut(row.hgvsc):
            gws_sample_ids.add(row.sample_id)
        # For SNVs, parse the ref and alt nucleotides, and then get the flanks
        if mut_type == 'SNV':
            num_snvs[row.sample_id] += 1
            ref, alt = parse_snv_edit(edit)
            # Get the bin index
            chrom_idx = get_chrom_idx_by_chrom(chrom)
            genomic_bin_idx = floor(int(pos) / bin_len) + (np.cumsum(num_bins)[chrom_idx-2] if chrom_idx > 1 else 0)
            seq = genomic_seqs[chrom]
            # Perform a sanity check
            if sanity_check(pos, ref, seq):
                f5, f3 = get_flanks(pos, seq)
                # Update the dictionary
                mut_dict['sample_id'].append(row.sample_id)
                mut_dict['mut_type'].append('SNV')
                mut_dict['gene'].append(revise_gene(row.gene))
                mut_dict['site'].append(row.site)
                mut_dict['chrom'].append(chrom)
                mut_dict['pos'].append(pos)
                mut_dict['bin'].append(genomic_bin_idx)
                mut_dict['f5'].append(f5)
                mut_dict['ref'].append(ref)
                mut_dict['alt'].append(alt)
                mut_dict['f3'].append(f3)
                mut_dict['mut_effect'].append(row.mut_effect.replace(' ', '').replace('-', ''))
                num_valid_rows += 1
                valid_sample_ids.add(row.sample_id)
            else:
                print('Expected: {}, but ref: {}'.format(nuc_at(seq, pos), ref))
                num_invalid_rows += 1
                invalid_sample_ids.add(row.sample_id)
        valid_sample_ids -= invalid_sample_ids

    print('Numbers of valid/invalid rows: {}/{}, Numbers of valid/invalid samples: {}/{}'.format(num_valid_rows, num_invalid_rows, len(valid_sample_ids), len(invalid_sample_ids)))

    min_num_mutations = inf
    max_num_mutations = -1
    sum_num_mutations = 0
    for sample_id in gws_sample_ids:
        n = num_snvs[sample_id]
        max_num_mutations = max(n, max_num_mutations)
        min_num_mutations = min(n, min_num_mutations)
        sum_num_mutations += n
    print('Average/Max/Min numbers of mutations of a single sample: {}/{}/{}'.format(sum_num_mutations/len(gws_sample_ids), max_num_mutations, min_num_mutations))

    # Only keep the samples which went through GWS
    final_snv_dict = defaultdict(list)
    for idx in range(len(mut_dict['sample_id'])):
        if mut_dict['sample_id'][idx] in gws_sample_ids:
            for key in mut_dict.keys():
                final_snv_dict[key].append(mut_dict[key][idx])

    return pd.DataFrame.from_dict(final_snv_dict)

def preprocess_cnv(df_mut, df_cnv):
    df_cnv = df_cnv.rename(cols_cnv_rename_mapper, axis=1)
    lb = len(df_cnv)
    criterion = df_cnv['sample_id'].map(lambda x: x in df_mut['sample_id'])
    df_cnv = df_cnv.loc[criterion]
    df_cnv['gene'] = df_cnv['gene'].apply(revise_gene)
    print('{} rows in the CNV file are remained out of {}'.format(len(df_cnv), lb)) 
    return df_cnv

def revise_gene(gene):
    if '_' in gene:
        gene = gene.split('_')[0]
    return gene

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Test with only a number of rows', type=int)
    parser.add_argument('--skip', nargs='+')
    args = parser.parse_args()
    n_testrows = args.test
    skip_files = args.skip
    # print(skip_files)

    raw_mut_filepath = './data/raw/CosmicGenomeScreensMutantExport.tsv'
    processed_mut_filepath = './data/processed/ProcessedGenomeScreensMutantExport.tsv'
    raw_cnv_filepath = './data/raw/CosmicCompleteCNA.tsv'
    processed_cnv_filepath = './data/processed/ProcessedCompleteCNA.tsv'

    print('Reading the mutation file')
    if 'mut' not in skip_files:    
        df_mut = pd.read_csv(raw_mut_filepath, sep='\t', encoding='ISO-8859-1', usecols=cols_mut, nrows=n_testrows)
        # Preprocess the raw mutation file
        print('Preprocess the mutation file')
        df_mut_processed = preprocess_mut(df_mut)
        # Store the processed mut file
        if not n_testrows:
            df_mut_processed.to_csv(processed_mut_filepath, sep="\t", index=False)
            print('The processed mutation file is stored in {}'.format(processed_mut_filepath))
    else:
        df_mut_processed = pd.read_csv(processed_mut_filepath, sep='\t', nrows=n_testrows)
    print(df_mut_processed.sample(n=10))
    
    if 'cnv' not in skip_files:
        print('Reading the CNV file')
        df_cnv = pd.read_csv(raw_cnv_filepath, sep='\t', encoding='ISO-8859-1', usecols=cols_cnv, nrows=n_testrows)
        # Preprocess the raw CNV file
        print('Preprocess the CNV file')
        df_cnv_processed = preprocess_cnv(df_mut_processed, df_cnv)
        # Store the processed CNV file
        if not n_testrows:
            df_cnv_processed.to_csv(processed_cnv_filepath, sep="\t", index=False)
            print('The processed CNV file is stored in {}'.format(processed_cnv_filepath))
        print(df_cnv_processed.head(10))

if __name__ == '__main__':
    main()