import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from math import ceil
from src.utils.hgvs import is_range, parse_hgvs, parse_ins_edit, parse_range
from src.utils.sequences import get_chrom_idx_by_chrom, get_flanks, get_genomic_sequences, seq_between
from src.utils.features import assign_indel_type, assign_sbs_type, assign_cnv_type, get_cnv_types, get_contextualized_sbs_types, get_indel_types, get_bin_idx, get_genomic_bins
from src.utils.classes_selection import get_interested_sites
import argparse
import os

# Get the mutational types
contextualized_sbs_types = get_contextualized_sbs_types()
# dbs_types = get_dbs_types()
indel_types = get_indel_types()
cnv_types = get_cnv_types()

# Get the genomic sequences
genomic_fasta_filepath = './data/external/GCF_000001405.26_GRCh38_genomic.fna'
genomic_seqs = get_genomic_sequences(genomic_fasta_filepath)
BIN_LEN = 1000000
genomic_bins = get_genomic_bins(genomic_seqs, bin_len=BIN_LEN)

with open('./data/processed/census_genes.pkl', 'rb') as f:
# with open('./data/processed/curated_genes.pkl', 'rb') as f:
    # census_gene_list = pickle.load(f)
    gene_list = pickle.load(f)
# df_oncokb = pd.read_csv('./data/external/oncokb_cancer_genes.tsv', sep='\t')
# df_oncokb = df_oncokb[df_oncokb['MSK-IMPACT'] == 'Yes']
# oncokb_gene_list = df_oncokb['Hugo Symbol'].to_list()
# gene_list = [gene for gene in oncokb_gene_list if gene in census_gene_list]

def normalized(vec):
    vec_sum = np.sum(vec)
    if vec_sum == 0:
        return vec
    else:
        return vec / vec_sum

def get_feat_vec(genomic_variants, copy_number_variants):
    # Mutational profile vectors
    sbs_mut_vec = np.zeros(len(contextualized_sbs_types))
    indel_mut_vec = np.zeros(len(indel_types))
    cnv_vec = np.zeros(len(cnv_types))
    # dbs_mut_vec = np.zeros(len(dbs_types))

    # Positional distribution vector
    sbs_pos_vec = np.zeros(np.sum(genomic_bins))
    indel_pos_vec = np.zeros(np.sum(genomic_bins))
    cnv_pos_vec = np.zeros(np.sum(genomic_bins))

    # Gene-level feature
    # gene_vec = np.zeros(len(gene_list))

    for var in genomic_variants:
        chrom, pos, edit, mut_type = parse_hgvs(var)
        seq = genomic_seqs[chrom]
        chrom_idx = get_chrom_idx_by_chrom(chrom)
        bin_idx = get_bin_idx(pos, chrom_idx, BIN_LEN, genomic_bins)
        
        if mut_type == 'SBS':
            # Mutational profile
            ref, alt = edit.split('>')
            f5, f3 = get_flanks(pos, seq)
            sbs_type = assign_sbs_type(ref, alt, f5, f3)
            if sbs_type in contextualized_sbs_types:
                sbs_mut_vec[contextualized_sbs_types.index(sbs_type)] += 1
            # Positional distribution
            sbs_pos_vec[bin_idx] += 1
        elif 'INDEL' in mut_type:
            if is_range(pos):
                pos_start, pos_end = parse_range(pos)
            else:
                pos_start = int(pos)
                pos_end = int(pos)
            subtype = mut_type.split('_')[-1]
            if subtype == 'DELINS' or subtype == 'DEL':
                del_edit = seq_between(seq, pos_start, pos_end)
                del_indel_type = assign_indel_type('DEL', del_edit, seq, pos_start, pos_end)
                indel_mut_vec[indel_types.index(del_indel_type)] += 1
            if subtype == 'DELINS' or subtype == 'INS':
                ins_edit = parse_ins_edit(edit)
                ins_indel_type = assign_indel_type('INS', ins_edit, seq, pos_start, pos_end)
                indel_mut_vec[indel_types.index(ins_indel_type)] += 1

            indel_pos_vec[bin_idx] += 1

    for tcn, coor in copy_number_variants:
    # for tcn, coor, minor in copy_number_variants:
        # Calculate the size
        start, end = coor.split(':')[-1].split('..')
        size = int(end) - int(start) + 1
        # Assign the CNV type
        cnv_type = assign_cnv_type(tcn, size)
        # cnv_type = assign_cnv_type(tcn, size, minor)

        if cnv_type in cnv_types:
            cnv_vec[cnv_types.index(cnv_type)] += 1
        else:
            print('CNV Type {} is unknown'.format(cnv_type))

        chrom = coor.split(':')[0]
        seq = genomic_seqs[chrom]

        chrom_idx = get_chrom_idx_by_chrom(chrom)
        bin_idx = get_bin_idx(int(start), chrom_idx, BIN_LEN, genomic_bins)

        cnv_pos_vec[bin_idx] += 1

    # for row in df_sample.itertuples():
    #     chrom, pos, edit, mut_type = parse_hgvs(row.hgvsg)
    #     chrom_idx = get_chrom_idx_by_chrom(chrom)
    #     seq = genomic_seqs[chrom]
    #     bin_idx = get_bin_idx(pos, chrom_idx, bin_len, genomic_bins)
    #     pos_vec[bin_idx] += 1
    #     if row.gene in gene_list:
    #         gene_vec[gene_list.index(row.gene)] += 1
    #     # Parse the mutation and update the corresponding feature vectors
    #     if mut_type == 'SBS':
    #         # Get the genomic bin index
    #         ref, alt = edit.split('>')
    #         f5, f3 = get_flanks(pos, seq)
    #         sbs_type = assign_sbs_type(ref, alt, f5, f3)
    #         if sbs_type in contextualized_sbs_types:
    #             sbs_mut_vec[contextualized_sbs_types.index(sbs_type)] += 1
    #     elif 'INDEL' in mut_type:
    #         if is_range(pos):
    #             pos_start, pos_end = parse_range(pos)
    #         else:
    #             pos_start = int(pos)
    #             pos_end = int(pos)
    #         subtype = mut_type.split('_')[-1]
    #         if subtype == 'DELINS' or subtype == 'DEL':
    #             del_edit = seq_between(seq, pos_start, pos_end)
    #             # print(pos_start, pos_end, del_edit)
    #             del_indel_type = assign_indel_type('DEL', del_edit, seq, pos_start, pos_end)
    #             indel_mut_vec[indel_types.index(del_indel_type)] += 1
    #         if subtype == 'DELINS' or subtype == 'INS':
    #             ins_edit = parse_ins_edit(edit)
    #             ins_indel_type = assign_indel_type('INS', ins_edit, seq, pos_start, pos_end)
    #             indel_mut_vec[indel_types.index(ins_indel_type)] += 1

    # feat_vec = normalized_concat([sbs_mut_vec, indel_mut_vec, pos_vec, gene_vec]).reshape(1, -1)

    # Normalize and concatenate into the feature vector

    feats = []
     
    feats.append(normalized(sbs_mut_vec))
    feats.append(normalized(indel_mut_vec))
    feats.append(normalized(cnv_vec))
    feats.append(normalized(sbs_pos_vec))
    feats.append(normalized(indel_pos_vec))
    feats.append(normalized(cnv_pos_vec))

    feat_vec = np.concatenate(feats)

    return feat_vec

def generate_features(sample_ids, feature_codename):
    mutation_dirpath = './data/interim/dataset/samples_mut/'
    cnv_dirpath = './data/interim/dataset/samples_cnv/'
    features_dirpath = './data/interim/features/' + feature_codename + '/'
    # Make the dir
    if not os.path.isdir(features_dirpath):
        os.mkdir(features_dirpath)
    # Generate the features for each sample
    for sid in tqdm(sample_ids):
        # Load the mutation file of the sample
        df_sample_mut = pd.read_csv(mutation_dirpath + str(sid) + '.tsv', sep='\t')
        df_sample_cnv = pd.read_csv(cnv_dirpath + str(sid) + '.tsv', sep='\t')
        # Take the genomic variants
        genomic_variants = df_sample_mut['hgvsg']
        copy_number_variants = zip(df_sample_cnv['total_cn'], df_sample_cnv['genomic_coordinates'], df_sample_cnv['minor'])
        # Generate the feature vector
        features = get_feat_vec(genomic_variants, copy_number_variants)
        # Save the features
        np.save(features_dirpath + str(sid) + '.npy', features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--codename', help='The codename of the feature version')
    feature_codename = parser.parse_args().codename

    # Get the interested sites
    sites_of_interest = get_interested_sites()
    # Load the dataset
    sites_df = pd.read_csv('./data/interim/dataset/sites.tsv', sep='\t')
    sample_ids = sites_df[sites_df['site'].map(lambda x: x in sites_of_interest)]['sample_id'].to_list()
    # Generate features
    print('Building the features')
    generate_features(sample_ids, feature_codename)

if __name__ == '__main__':
    main()