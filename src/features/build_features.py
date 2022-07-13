# TODO: Add header comments

# from ast import literal_eval
# from collections import defaultdict
import numpy as np
import pickle
from torch import norm
from tqdm import tqdm
from itertools import product
# from src.utils.hgvs_parsing import is_range
from src.utils.sequences import assign_mut_type

# bin_length = 1000000 # 1 million
# num_of_chromosomes = 22

sub_types = [('C', x) for x in 'AGT'] + [('T', x) for x in 'ACG']
# context_types = [x for x in product('[ACGT', sub_types, 'ACGT]')]
context_types = [x for x in product('ACGT', sub_types, 'ACGT')]
with open('./data/processed/census_genes.pkl', 'rb') as f:
# with open('./data/processed/curated_genes.pkl', 'rb') as f:
    gene_list = pickle.load(f)

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

def get_snv_vector(sample_id, dataset):
    gene_vec = np.zeros(len(gene_list))
    distro_vec = np.zeros(3102) # FIXME: hard-coded
    context_type_vec = np.zeros(len(context_types))

    for gene, bin, f5, ref, alt, f3 in dataset['snv_dict'][sample_id]:
        context_type = assign_mut_type(ref, alt, f5, f3)
        mut_type_idx = context_types.index(context_type)
        context_type_vec[mut_type_idx] += 1
        if gene in gene_list:
            gene_vec[gene_list.index(gene)] += 1
        distro_vec[bin] += 1

    return normalized_concat([gene_vec, distro_vec, context_type_vec]).reshape(1, -1)

def get_cnv_vector(sample_id, dataset):
    cnv_vec = np.zeros(len(gene_list) * 2)

    for gene, total_cn, mut_type in dataset['cnv_dict'][sample_id]:
        if gene in gene_list:
            gene_idx = gene_list.index(gene)
            cnv_vec[1 + gene_idx + 0 if mut_type == 'gain' else 1] = total_cn
    
    return cnv_vec.reshape(1, -1)

def get_feature_matrix(dataset):
    feat_matrix = []
    # bin_counts = calculate_bin_counts()
    # for sample_id in tqdm(dataset['sample_ids']):
    for sample_id in tqdm(dataset['sample_ids']):
        snv_vec = get_snv_vector(sample_id, dataset)
        # cnv_vec  = get_cnv_vector(sample_id, dataset)
        # feat_vec = np.concatenate([cmut_vec, cnv_vec], axis=1)
        feat_vec = snv_vec
        feat_matrix.append(feat_vec)
    feat_matrix = np.concatenate(feat_matrix, axis=0, dtype=np.float32)
    print('Non-zero count:', np.count_nonzero(feat_matrix)) # FIXME
    # print(feat_matrix)
    return feat_matrix

def main():
    # Load the dataset
    with open('./data/interim/dataset.pkl', 'rb') as f: # FIXME
        dataset = pickle.load(f)
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