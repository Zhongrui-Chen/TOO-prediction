import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from math import ceil
from src.utils.hgvs_parsing import is_range, parse_hgvs, parse_range
from src.utils.sequences import get_chrom_by_idx, get_chrom_idx_by_chrom, get_flanks, get_genomic_sequences, seq_between
from src.utils.features import assign_indel_type, assign_sbs_type, get_contextualized_sbs_types, get_indel_types, get_bin_idx

indel_range_bound = 10
contextualized_sbs_types = get_contextualized_sbs_types()
indel_types = get_indel_types(indel_range_bound)

# Get the genomic sequences
genomic_fasta_filepath = './data/external/GCF_000001405.26_GRCh38_genomic.fna'
num_chromosomes = 24
bin_len = 1000000 # 1 million
genomic_seqs = get_genomic_sequences(genomic_fasta_filepath)
# Calculate the numbers of bins in each chromosome
num_bins = np.zeros(num_chromosomes, dtype=int)
for chrom_idx in range(1, num_chromosomes+1):
    chrom = get_chrom_by_idx(chrom_idx)
    num_bins[chrom_idx-1] = ceil(len(genomic_seqs[chrom]) / bin_len)
total_num_bins = np.sum(num_bins)
print('Total number of bins is', total_num_bins)

with open('./data/processed/census_genes.pkl', 'rb') as f:
# with open('./data/processed/curated_genes.pkl', 'rb') as f:
    # census_gene_list = pickle.load(f)
    gene_list = pickle.load(f)
# df_oncokb = pd.read_csv('./data/external/oncokb_cancer_genes.tsv', sep='\t')
# df_oncokb = df_oncokb[df_oncokb['MSK-IMPACT'] == 'Yes']
# oncokb_gene_list = df_oncokb['Hugo Symbol'].to_list()
# gene_list = [gene for gene in oncokb_gene_list if gene in census_gene_list]

def normalized(vec):
    if np.linalg.norm(vec) != 0:
        return vec / np.linalg.norm(vec)
    else:
        return vec

def normalized_concat(vecs):
    concat_vecs = []
    for v in vecs:
        concat_vecs.append(normalized(v))
    return np.concatenate(concat_vecs)

def get_feat_vec(mutation_dirpath, sample_id):
    # Load the mutation file of the sample
    df_sample = pd.read_csv(mutation_dirpath + str(sample_id) + '.tsv', sep='\t')

    sbs_mut_vec = np.zeros(len(contextualized_sbs_types))
    indel_mut_vec = np.zeros(len(indel_types))
    pos_vec = np.zeros(total_num_bins)
    # sbs_pos_vec = np.zeros(total_num_bins)
    # indel_pos_vec = np.zeros(total_num_bins)
    # dup_pos_vec = np.zeros(total_num_bins)
    sbs_gene_vec = np.zeros(len(gene_list))
    indel_gene_vec = np.zeros(len(gene_list))
    dup_gene_vec = np.zeros(len(gene_list))

    for row in df_sample.itertuples():
        chrom, ref_type, pos, edit, mut_type = parse_hgvs(row.hgvsg)
        if ref_type != 'g':
            continue
        # Get the genomic bin index
        chrom_idx = get_chrom_idx_by_chrom(chrom)
        seq = genomic_seqs[chrom]
        bin_idx = get_bin_idx(pos, chrom_idx, bin_len, num_bins)
        pos_vec[bin_idx] += 1
        # Parse the mutation and update the corresponding feature vectors
        if mut_type == 'SBS':
            ref, alt = edit.split('>')
            f5, f3 = get_flanks(pos, seq)
            sbs_type = assign_sbs_type(ref, alt, f5, f3)
            sbs_mut_vec[contextualized_sbs_types.index(sbs_type)] += 1
            if row.gene in gene_list:
                sbs_gene_vec[gene_list.index(row.gene)] += 1
            # sbs_pos_vec[bin_idx] += 1
        elif 'INDEL' in mut_type:
            if is_range(pos):
                pos_start, pos_end = parse_range(pos)
            else:
                pos_start = int(pos)
                pos_end = int(pos)
            subtype = mut_type.split('_')[-1]
            del_length = pos_end - pos_start + 1
            ins_length = len(edit)
            if subtype == 'DELINS' or subtype == 'DEL':
                del_edit = seq_between(seq, pos_start, pos_end)
                del_indel_type = assign_indel_type('DEL', del_length, del_edit, indel_range_bound)
                indel_mut_vec[indel_types.index(del_indel_type)] += 1
                if row.gene in gene_list:
                    indel_gene_vec[gene_list.index(row.gene)] += 1
                # indel_pos_vec[bin_idx] += 1
            if subtype == 'DELINS' or subtype == 'INS':
                ins_indel_type = assign_indel_type('INS', ins_length, edit, indel_range_bound)
                indel_mut_vec[indel_types.index(ins_indel_type)] += 1
                # indel_pos_vec[bin_idx] += 1
                if row.gene in gene_list:
                    indel_gene_vec[gene_list.index(row.gene)] += 1
        # elif 'DUP' in mut_type:
        #     if row.gene in gene_list:
        #         dup_gene_vec[gene_list.index(row.gene)] += 1

    return normalized_concat([sbs_mut_vec, indel_mut_vec, pos_vec, sbs_gene_vec, indel_gene_vec, dup_gene_vec]).reshape(1, -1)

def get_feature_matrix(mutation_dirpath, sample_ids):
    feat_matrix = []
    for sample_id in tqdm(sample_ids):
        feat_vec = get_feat_vec(mutation_dirpath, sample_id)
        feat_matrix.append(feat_vec)
    feat_matrix = np.concatenate(feat_matrix, axis=0, dtype=np.float32)
    print('Non-zero count:', np.count_nonzero(feat_matrix)) # FIXME
    return feat_matrix

def main():
    # Load the dataset
    samples_df = pd.read_csv('./data/interim/dataset/sites.tsv', sep='\t')
    sample_ids = samples_df['sample_id'].to_list()
    # Generate a feature vector for each sample to constrcut the feature matrix
    mutation_dirpath = './data/interim/dataset/samples/'
    print('Building the feature matrix')
    fmatrix = get_feature_matrix(mutation_dirpath, sample_ids)
    print('The shape of feature matrix is {}'.format(fmatrix.shape))
    # Save the feature matrix
    feature_filepath = './data/interim/features/features.npy'
    np.save(feature_filepath, fmatrix)
    print('The feature matrix is stored in {}'.format(feature_filepath))

if __name__ == '__main__':
    main()