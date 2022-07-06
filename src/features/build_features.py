# TODO: Add header comments

from collections import defaultdict
from random import sample
import numpy as np
import pickle
from tqdm import tqdm
from itertools import product
# from src.utils.hgvs_parsing import parse_cmut
# from src.utils.sequences import get_cds_lookup_table, reverse_complement, get_reference_by_gene_and_accession_number, seq_between, nuc_at
# from src.utils.sequences import get_cds_lookup_table, get_flanks
# from collections import defaultdict

# bin_length = 1000000 # 10 million
# num_of_chromosome = 22
# vocab = [''.join(p) for p in itertools.product('ATGC', repeat=3)]

sub_types = [('C', x) for x in 'AGT'] + [('T', x) for x in 'ACG']
mut_types = [x for x in product('<ACGT', sub_types, 'ACGT>')]
with open('./data/processed/census_genes.pkl', 'rb') as f:
# with open('./data/processed/curated_genes.pkl', 'rb') as f:
    gene_list = pickle.load(f)

class UnknownMutTypeException(Exception):
    pass

# def calculate_bin_counts():
#     '''
#     Calculate how many bins should be allocated for each chromosome
#     '''
#     print('[Calculating the bins for mutation distribution]')
#     bin_counts = {}
#     for (_, chrom), gmuts in dataset['gmut_dict'].items():
#         for mut in gmuts:
#             # var = hp.parse_hgvs_variant(mut)
#             # pos = var.posedit.pos
#             seg = mut.split('.')[1]
#             pos = ''
#             for ch in seg:
#                 if not ch.isalpha():
#                     pos += ch
#                 else:
#                     break
#             if is_range(pos):
#                 # pos = pos.start
#                 pos = pos.split('_')[0]
#             bin_idx = int(int(str(pos)) / bin_length)
#             if chrom in bin_counts:
#                 if bin_idx > bin_counts[chrom]:
#                     bin_counts[chrom] = bin_idx
#             else:
#                 bin_counts[chrom] = bin_idx
#     bin_counts_list = np.zeros(num_of_chromosome, dtype=int)
#     for chrom in range(1, num_of_chromosome + 1):
#         chrom_str = str(chrom)
#         bin_counts_list[chrom - 1] = bin_counts[chrom_str]
#     return bin_counts_list

# def get_gmut_distribution_vector(sample_id, bin_counts):
#     bin_counts_cumsum = np.cumsum(bin_counts)
#     distro_vec = np.zeros(np.sum(bin_counts))
#     for chrom in range(1, num_of_chromosome + 1):
#         chrom_str = str(chrom)
#         if (sample_id, chrom_str) in dataset['gmut_dict']:
#             for mut in dataset['gmut_dict'][(sample_id, chrom_str)]:
#                 # var = hp.parse_hgvs_variant(mut)
#                 # pos = var.posedit.pos
#                 seg = mut.split('.')[1]
#                 pos = ''
#                 for ch in seg:
#                     if not ch.isalpha():
#                         pos += ch
#                     else:
#                         break
#                 if is_range(pos):
#                     # pos = pos.start
#                     pos = pos.split('_')[0]
#                 bin_idx = int(int(str(pos)) / bin_length)
#                 distro_vec[bin_counts_cumsum[chrom - 2] if chrom > 1 else 0 + bin_idx] += 1
#     norm_factor = np.sum(distro_vec) if np.sum(distro_vec) > 0 else 1
#     return (distro_vec / norm_factor).reshape((1, -1))

# def parse_patterns_of_var(var, padded_seq, margin):
#     mut_type = parse_mut_type(var)
#     # Pad sequence at ends
#     if mut_type == 'sub':
#         pos = int(str(var.posedit.pos)) + margin
#         alt = var.posedit.edit.alt
#         ref_region_start = pos - margin
#         ref_region_end = pos + 1
#         alt_region_start = ref_region_start
#         alt_region_end = ref_region_end
#         altered_seq = seq_between(padded_seq, 1, pos-1) + alt + seq_between(padded_seq, pos+1, len(padded_seq))
#     elif mut_type == 'del':
#         # It can be a single position or a range
#         if not is_range(var.posedit.pos):
#             pos = int(str(var.posedit.pos)) + margin
#             ref_region_start = pos - margin
#             ref_region_end = pos + 1
#             alt_region_start = ref_region_start
#             alt_region_end = ref_region_end - 1
#             altered_seq = seq_between(padded_seq, 1, pos-1) + seq_between(padded_seq, pos+1, len(padded_seq))
#         else:
#             pos_start = int(str(var.posedit.pos.start)) + margin 
#             pos_end = int(str(var.posedit.pos.end)) + margin
#             ref_region_start = pos_start - margin
#             ref_region_end = pos_end + 1
#             alt_region_start = ref_region_start
#             alt_region_end = ref_region_end - (pos_end - pos_start + 1)
#             altered_seq = seq_between(padded_seq, 1, pos_start-1) + seq_between(padded_seq, pos_end+1, len(padded_seq))
#     elif mut_type == 'dup':
#         if not is_range(var.posedit.pos):
#             pos = int(str(var.posedit.pos)) + margin
#             alt = nuc_at(padded_seq, pos)
#             ref_region_start = pos - margin
#             ref_region_end = pos + 1
#             alt_region_start = ref_region_start
#             alt_region_end = ref_region_end + 1
#             altered_seq = seq_between(padded_seq, 1, pos-1) + alt * 2 + seq_between(padded_seq, pos+1, len(padded_seq))
#         else:
#             pos_start = int(str(var.posedit.pos.start)) + margin 
#             pos_end = int(str(var.posedit.pos.end)) + margin
#             alt = seq_between(padded_seq, pos_start, pos_end)
#             ref_region_start = pos_start - margin
#             ref_region_end = pos_end + 1
#             alt_region_start = ref_region_start
#             alt_region_end = ref_region_end + (pos_end - pos_start + 1)
#             altered_seq = seq_between(padded_seq, 1, pos_start-1) + alt * 2 + seq_between(padded_seq, pos_end+1, len(padded_seq))
#     elif mut_type == 'ins':
#         # It must be two neighbouring flanking nucleotides, e.g., 123_124 but not 123_125
#         pos_start = int(str(var.posedit.pos.start)) + margin
#         pos_end = int(str(var.posedit.pos.end)) + margin
#         alt = str(var.posedit.edit.alt)
#         ref_region_start = pos_start - margin + 1
#         ref_region_end = pos_start + 1
#         alt_region_start = ref_region_start
#         alt_region_end = ref_region_end + len(alt)
#         altered_seq = seq_between(padded_seq, 1, pos_start) + alt + seq_between(padded_seq, pos_end, len(padded_seq))
#     elif mut_type == 'inv':
#         pos_start = int(str(var.posedit.pos.start)) + margin
#         pos_end = int(str(var.posedit.pos.end)) + margin
#         alt = ''.join([reverse_complement(nuc) for nuc in seq_between(padded_seq, pos_start, pos_end)])
#         ref_region_start = pos_start - margin
#         ref_region_end = pos_end + 1
#         alt_region_start = ref_region_start
#         alt_region_end = ref_region_end + 1
#         altered_seq = seq_between(padded_seq, 1, pos_start-1) + alt + seq_between(padded_seq, pos_end+1, len(padded_seq))
#     elif mut_type == 'delins':
#         alt = var.posedit.edit.alt
#         if not is_range(var.posedit.pos):
#             pos = int(str(var.posedit.pos)) + margin    
#             ref_region_start = pos - margin
#             ref_region_end = pos + 1
#             alt_region_start = ref_region_start
#             alt_region_end = ref_region_end + len(alt) - 1
#             altered_seq = seq_between(padded_seq, 1, pos-1) + alt + seq_between(padded_seq, pos+1, len(padded_seq))
#         else:
#             pos_start = int(str(var.posedit.pos.start)) + margin 
#             pos_end = int(str(var.posedit.pos.end)) + margin
#             ref_region_start = pos_start - margin
#             ref_region_end = pos_end + 1
#             alt_region_start = ref_region_start
#             alt_region_end = ref_region_end + (len(alt) - (pos_end - pos_start + 1))
#             altered_seq = seq_between(padded_seq, 1, pos_start-1) + alt + seq_between(padded_seq, pos_end+1, len(padded_seq))
#     else:
#         raise UnknownMutTypeException('Unknown mutation {} cannot be parsed'.format(var))
#     return altered_seq, ref_region_start, ref_region_end, alt_region_start, alt_region_end

# def get_cmut_freq_vector(sample_id):
#     norm_factor = 0
#     for gene_idx, gene in enumerate(dataset['genes']):
#         if (sample_id, gene) in dataset['cmut_dict']:
#             for mut in dataset['cmut_dict'][(sample_id, gene)]:
#                 var = hp.parse_hgvs_variant(mut)
#                 ac = var.ac
#                 seq_ref = get_reference_by_gene_and_accession_number(gene, ac, cds_lookup_table)
#                 seq = cds_lookup_table.fetch(seq_ref).upper()
#                 pad = '='
#                 margin = k_mers - 1
#                 padded_seq = margin * pad + seq + margin * pad
#                 altered_seq, ref_region_start, ref_region_end, alt_region_start, alt_region_end = parse_patterns_of_var(var, padded_seq, margin)
#                 # Collect old and new patterns
#                 for idx in range(ref_region_start, ref_region_end):
#                     mer = seq_between(padded_seq, idx, idx+margin)
#                     if '=' in mer or '' == mer:
#                         continue
#                     mer_idx = vocab.index(mer)
#                     # pattern_vec[gene_idx * len(vocab) + mer_idx] -= 1
#                     pattern_vec[mer_idx] -= 1
#                     norm_factor += 1
#                 for idx in range(alt_region_start, alt_region_end):
#                     mer = seq_between(altered_seq, idx, idx+margin)
#                     if '=' in mer or '' == mer:
#                         continue
#                     mer_idx = vocab.index(mer)
#                     # pattern_vec[gene_idx * len(vocab) + mer_idx] += 1
#                     pattern_vec[mer_idx] += 1
#                     norm_factor += 1
#     if norm_factor == 0:
#         norm_factor = 1
#     return (pattern_vec / norm_factor).reshape((1, -1))

def get_cmut_vector(sample_id, dataset):
    # mut_gene_dict = dict.fromkeys(gene_list, 0)
    # mut_distro_dict = dict.fromkeys(mut_types, 0)

    # print(len(mut_freq_dict))

    mut_gene_vec = np.zeros(len(gene_list))
    mut_type_vec = np.zeros(len(mut_types))

    for gene, mut_type in dataset['cmut_dict'][sample_id]:
        mut_type_idx = mut_types.index(mut_type)
        mut_type_vec[mut_type_idx] += 1
        if gene in gene_list:
            gene_idx = gene_list.index(gene)
            mut_gene_vec[gene_idx] = 1
    
    mut_type_vec /= np.sum(mut_type_vec) if np.sum(mut_type_vec) > 0 else 1
    mut_vec = np.concatenate((mut_type_vec, mut_gene_vec))

    return mut_vec.reshape(1, -1)

def get_cnv_vector(sample_id, dataset):
    cnv_vec = np.zeros(len(gene_list) * 2)

    for gene, mut_type in dataset['cnv_dict'][sample_id]:
        if gene in gene_list:
            gene_idx = gene_list.index(gene)
            cnv_vec[1 + gene_idx + 0 if mut_type == 'gain' else 1] = 1
    
    return cnv_vec.reshape(1, -1)

# def get_ge_vector(sample_id, dataset):
#     ge_vec = np.zeros(len(gene_list))
#     for gene, z_score in dataset['ge_dict'][sample_id]:
#         if gene in gene_list:
#             gene_idx = gene_list.index(gene)
#             ge_vec[gene_idx] = z_score
#     return ge_vec.reshape(1, -1)py

def get_feature_matrix(dataset):
    feat_matrix = []
    # bin_counts = calculate_bin_counts()
    for sample_id in tqdm(dataset['sample_ids']):
    # test_idcs = np.random.randint(0, len(dataset['sample_ids']), 5)
    # test_sample_ids = [dataset['sample_ids'][idx] for idx in test_idcs]
    # for sample_id in tqdm(test_sample_ids):
        # distro_vec  = get_gmut_distribution_vector(sample_id, bin_counts)
        cmut_distro_vec = get_cmut_vector(sample_id, dataset)
        cnv_vec = get_cnv_vector(sample_id, dataset)
        # ge_vec = get_ge_vector(sample_id, dataset)
        feat_vec = np.concatenate((cmut_distro_vec, cnv_vec), axis=1)
        feat_matrix.append(feat_vec)
    feat_matrix = np.concatenate(feat_matrix, axis=0, dtype=np.float32)
    print('Non-zero count:', np.count_nonzero(feat_matrix)) # FIXME
    # print(feat_matrix)
    return feat_matrix

def main():
    # Load the dataset
    with open('./data/interim/dataset.pkl', 'rb') as f: # FIXME
        dataset = pickle.load(f)
    # Test FIXME only use a portion of data
    # Generate a feature vector for each sample to constrcut the feature matrix
    print('Building the feature matrix')
    fmatrix = get_feature_matrix(dataset)
    print('The shape of feature matrix is {}'.format(fmatrix.shape))
    # Save the feature matrix
    feature_filepath = './data/interim/features/features.npy'
    np.save(feature_filepath, fmatrix)
    print('The feature matrix is stored in {}'.format(feature_filepath))

if __name__ == '__main__':
    main()