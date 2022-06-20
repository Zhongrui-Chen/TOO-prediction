import logging
import pandas as pd
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import itertools
import hgvs.parser
from src.data import cds_lookup
from src.data.quality_control import filter_dataset
from src.utils.uhgvs import parse_mut_type
from src.utils.useq import get_reference_by_gene_and_accession_number, seq_between, nuc_at
# cleansed_export_filepath = './data/processed/CleansedMutantExportCensus.tsv'

class UnknownMutTypeException(Exception):
    pass

def is_range(pos):
    return '_' in str(pos)

def reverse_complement(nuc):
    if nuc == 'A':
        return 'G'
    elif nuc == 'T':
        return 'C'
    elif nuc == 'G':
        return 'A'
    elif nuc == 'C':
        return 'T'

def parse_local_features_of_var(var, padded_seq, margin):
    mut_type = parse_mut_type(var)
    # Pad sequence at ends
    if mut_type == 'sub':
        pos = int(str(var.posedit.pos)) + margin
        alt = var.posedit.edit.alt
        ref_region_start = pos - margin
        ref_region_end = pos + 1
        alt_region_start = ref_region_start
        alt_region_end = ref_region_end
        altered_seq = seq_between(padded_seq, 1, pos-1) + alt + seq_between(padded_seq, pos+1, len(padded_seq))
    elif mut_type == 'del':
        # It can be a single position or a range
        if not is_range(var.posedit.pos):
            pos = int(str(var.posedit.pos)) + margin
            ref_region_start = pos - margin
            ref_region_end = pos + 1
            alt_region_start = ref_region_start
            alt_region_end = ref_region_end - 1
            altered_seq = seq_between(padded_seq, 1, pos-1) + seq_between(padded_seq, pos+1, len(padded_seq))
        else:
            pos_start = int(str(var.posedit.pos.start)) + margin 
            pos_end = int(str(var.posedit.pos.end)) + margin
            ref_region_start = pos_start - margin
            ref_region_end = pos_end + 1
            alt_region_start = ref_region_start
            alt_region_end = ref_region_end - (pos_end - pos_start + 1)
            altered_seq = seq_between(padded_seq, 1, pos_start-1) + seq_between(padded_seq, pos_end+1, len(padded_seq))
    elif mut_type == 'dup':
        if not is_range(var.posedit.pos):
            pos = int(str(var.posedit.pos)) + margin
            alt = nuc_at(padded_seq, pos)
            ref_region_start = pos - margin
            ref_region_end = pos + 1
            alt_region_start = ref_region_start
            alt_region_end = ref_region_end + 1
            altered_seq = seq_between(padded_seq, 1, pos-1) + alt * 2 + seq_between(padded_seq, pos+1, len(padded_seq))
        else:
            pos_start = int(str(var.posedit.pos.start)) + margin 
            pos_end = int(str(var.posedit.pos.end)) + margin
            alt = seq_between(padded_seq, pos_start, pos_end)
            ref_region_start = pos_start - margin
            ref_region_end = pos_end + 1
            alt_region_start = ref_region_start
            alt_region_end = ref_region_end + (pos_end - pos_start + 1)
            altered_seq = seq_between(padded_seq, 1, pos_start-1) + alt * 2 + seq_between(padded_seq, pos_end+1, len(padded_seq))
    elif mut_type == 'ins':
        # It must be two neighbouring flanking nucleotides, e.g., 123_124 but not 123_125
        pos_start = int(str(var.posedit.pos.start)) + margin
        pos_end = int(str(var.posedit.pos.end)) + margin
        alt = str(var.posedit.edit.alt)
        ref_region_start = pos_start - margin + 1
        ref_region_end = pos_start + 1
        alt_region_start = ref_region_start
        alt_region_end = ref_region_end + len(alt)
        altered_seq = seq_between(padded_seq, 1, pos_start) + alt + seq_between(padded_seq, pos_end, len(padded_seq))
    elif mut_type == 'inv':
        pos_start = int(str(var.posedit.pos.start)) + margin
        pos_end = int(str(var.posedit.pos.end)) + margin
        alt = ''.join([reverse_complement(nuc) for nuc in seq_between(padded_seq, pos_start, pos_end)])
        ref_region_start = pos_start - margin
        ref_region_end = pos_end + 1
        alt_region_start = ref_region_start
        alt_region_end = ref_region_end + 1
        altered_seq = seq_between(padded_seq, 1, pos_start-1) + alt + seq_between(padded_seq, pos_end+1, len(padded_seq))
    elif mut_type == 'delins':
        alt = var.posedit.edit.alt
        if not is_range(var.posedit.pos):
            pos = int(str(var.posedit.pos)) + margin    
            ref_region_start = pos - margin
            ref_region_end = pos + 1
            alt_region_start = ref_region_start
            alt_region_end = ref_region_end + len(alt) - 1
            altered_seq = seq_between(padded_seq, 1, pos-1) + alt + seq_between(padded_seq, pos+1, len(padded_seq))
        else:
            pos_start = int(str(var.posedit.pos.start)) + margin 
            pos_end = int(str(var.posedit.pos.end)) + margin
            ref_region_start = pos_start - margin
            ref_region_end = pos_end + 1
            alt_region_start = ref_region_start
            alt_region_end = ref_region_end + (len(alt) - (pos_end - pos_start + 1))
            altered_seq = seq_between(padded_seq, 1, pos_start-1) + alt + seq_between(padded_seq, pos_end+1, len(padded_seq))
    else:
        raise UnknownMutTypeException('Unknown mutation {} cannot be parsed'.format(var))
    return altered_seq, ref_region_start, ref_region_end, alt_region_start, alt_region_end

# def get_feature_matrix(tumour_ids, all_genes, mut_dict, k):
def get_feature_matrix(tumour_ids, dataset_dict, k):
    ''' Create the N * 716 * 64 feature matrix '''
    # tumour_ids = dataset_dict['qualified_tumour_ids']
    all_genes = dataset_dict['genes']
    mut_dict = dataset_dict['mut_dict']
    hp = hgvs.parser.Parser()
    lookup_table = cds_lookup.get_cds_lookup_table()
    k_mers_vocab = [''.join(p) for p in itertools.product('ATGC', repeat=k)]
    fmatrix = np.zeros((len(tumour_ids), len(all_genes), len(k_mers_vocab)), dtype=np.float32)
    # logger.info('Building the feature matrix')
    for tumour_idx, tumour_id in enumerate(tqdm(tumour_ids)):
        for gene_idx, gene in enumerate(all_genes):
            if (tumour_id, gene) in mut_dict:
                for mut in mut_dict[(tumour_id, gene)]:
                    var = hp.parse_hgvs_variant(mut)
                    # mut_type = parse_mut_type(var)
                    ac = var.ac
                    seq_ref = get_reference_by_gene_and_accession_number(gene, ac, lookup_table)
                    seq = lookup_table.fetch(seq_ref).upper()
                    pad = '='
                    margin = k - 1
                    padded_seq = margin * pad + seq + margin * pad
                    altered_seq, ref_region_start, ref_region_end, alt_region_start, alt_region_end = parse_local_features_of_var(var, padded_seq, margin)
                    # Collect old and new patterns
                    for idx in range(ref_region_start, ref_region_end):
                        mer = seq_between(padded_seq, idx, idx+margin)
                        if '=' in mer or '' == mer:
                            continue
                        mer_idx = k_mers_vocab.index(mer)
                        fmatrix[tumour_idx, gene_idx, mer_idx] -= 1
                    for idx in range(alt_region_start, alt_region_end):
                        mer = seq_between(altered_seq, idx, idx+margin)
                        if '=' in mer or '' == mer:
                            continue
                        mer_idx = k_mers_vocab.index(mer)
                        fmatrix[tumour_idx, gene_idx, mer_idx] += 1
    return fmatrix

def generate_feature_npy(dataset_dict, q, k):
    tumour_ids = filter_dataset(dataset_dict, q)
    fmatrix = get_feature_matrix(tumour_ids, dataset_dict, k)
    filepath = './data/interim/features/features_q={}_k={}.npy'.format(q, k)
    np.save(filepath, fmatrix)
    return filepath

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--q', help='quality threshold', type=int)
    argparser.add_argument('--k', help='k of k-mers', type=int)
    args = argparser.parse_args()
    q = args.q if args.q else 5
    k = args.k if args.k else 3
    logger = logging.getLogger(__name__)
    # Get the feature matrix
    # with open('./config.json', 'r') as f:
    #     config = json.load(f)
    # k_mers = config['kMers']
    with open('./data/interim/dataset_dict.pkl', 'rb') as f:
        dataset_dict = pickle.load(f)
    # logger.info('The feature matrix has been created with the shape {}'.format(fmatrix.shape))
    logger.info('Generating features for q = {} and k = {}'.format(q, k))
    filepath = generate_feature_npy(dataset_dict, q, k) 
    logger.info('The feature matrix is stored into {}'.format(filepath))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()