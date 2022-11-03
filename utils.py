import numpy as np
import pysam
# from math import ceil

def get_sites(key):
    d = {
        'all': ['blood', 'bone', 'brain', 'breast', 'colorect', 'esophageal', 'head_neck', 'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skin', 'stomach'],
        'inter': ['bone', 'brain', 'breast', 'colorect', 'esophageal', 'lung', 'pancreas', 'prostate', 'skin', 'stomach'],
        'enough': ['breast', 'pancreas', 'prostate', 'brain', 'esophageal', 'bone'],
        'f': ['breast', 'pancreas', 'prostate', 'brain', 'esophageal', 'bone', 'colorect'],
    }
    return d[key]

def seq_between(seq, start, end):
    # Included, e.g., ('ATGCA', 2, 4) => 'TGC'
    return seq[start-1 : end]

def get_genomic_sequences(fasta_filepath):
    tb = pysam.FastaFile(fasta_filepath)
    chrom_refs = [ac for ac in tb.references if 'NC' in ac]
    # print(chrom_refs)
    gseqs = {}
    for chrom_idx in range(1, 25):
        chrom_ref = chrom_refs[chrom_idx - 1]
        chrom = get_chrom_by_idx(chrom_idx)
        gseqs[chrom] = tb.fetch(chrom_ref).upper()
    return gseqs

# def get_genomic_bins(genomic_seqs, num_chromosomes=24, bin_len=1000000):
    # Calculate the numbers of bins in each chromosome
    # num_bins = np.zeros(num_chromosomes, dtype=int)
    # for chrom_idx in range(1, num_chromosomes+1):
    #     chrom = get_chrom_by_idx(chrom_idx)
    #     num_bins[chrom_idx-1] = ceil(len(genomic_seqs[chrom]) / bin_len)
    # return num_bins
    
def get_genomic_bins(num_chromosomes=24):
    bins = [250, 244, 199, 192, 181, 172, 160, 147, 142, 136, 136, 134, 116, 108, 103, 91, 82, 79, 60, 64, 49, 52, 156, 60]
    return bins[:num_chromosomes]

def get_chrom_by_idx(chrom_idx):
    if chrom_idx > 22:
        chrom = 'X' if chrom_idx == 23 else 'Y'
    else:
        chrom = str(chrom_idx)
    return chrom

def nuc_at(seq, pos):
    return seq[int(pos) - 1]

def get_complement(nuc):
    mapping = {
        'G': 'C',
        'C': 'G',
        'A': 'T',
        'T': 'A'
    }
    return mapping[nuc]

def normalized(vec):
    vec_sum = np.sum(vec)
    if vec_sum == 0:
        return vec
    else:
        return vec / vec_sum