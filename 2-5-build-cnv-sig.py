from math import inf, isinf
from itertools import product
import pandas as pd
import numpy as np
import os
import pysam
from tqdm import tqdm
from copy import copy

from lib.utils import get_complement, get_sites, nuc_at, normalized, get_genomic_sequences, seq_between

def get_size_intervals():
    s = [
        # 0, 1e3, 1e4, 1e5, 1e6, 1e7, 4e7
        0, 1e5, 1e6, 1e7, 4e7
    ]
    cnt = 96
    while True:
        prev = copy(s)
        for idx in range(len(prev) - 1):
            lower, upper = prev[idx], prev[idx+1]
            m = int((lower + upper) / 2)
            s.append(m)
            if len(s) >= cnt:
                break
        s = sorted(s)
        if len(s) >= cnt:
            break
    s.append(inf)

    return s
#     l = 416667
#     s = []


def get_cnv_types(size_intervals):
    cnv_types = []
    # size_categories = ['<=1k', '1k-5k', '5k-10k', '10k-50k', '50k-100k', '100k-500k', '500k-1M', '1M-5M', '5M-10M', '10M-40M', '>40M']
    for idx in range(len(size_intervals) - 1):
        lower, upper = size_intervals[idx], size_intervals[idx+1]
        cnv_types.append((lower, upper))
    return cnv_types

# def assign_size_category(size):
#     for idx, (lo, up) in enumerate(cnv_types):
#         if size > lo and size <= up:
#             return idx

cnv_types = get_cnv_types(get_size_intervals())
print(len(cnv_types))
print(cnv_types)

def assign_cnv_type(size):
    for idx, (lo, up) in enumerate(cnv_types):
        if size > lo and size <= up:
            return idx

# genomic_fasta_path = './data/GCF_000001405.25_GRCh37.p13_genomic.fna'
# genomic_seqs = get_genomic_sequences(genomic_fasta_path)

def get_cnv_sig(df):
    sig = np.zeros(len(cnv_types))

    for chrom_start, chrom_end in zip(df['chrom_start'], df['chrom_end']):
        chrom_start = int(chrom_start)
        chrom_end   = int(chrom_end)
        size = chrom_end - chrom_start + 1
        sig[assign_cnv_type(size)] += 1

    return normalized(sig)

def main():
    sites = get_sites('f')
    src_path = './data/ICGC-dataset/cnv'
    dst_path = './features/cnv-sig'

    # Get the table
    ftab = pd.read_csv('./data/ICGC-dataset/sample-table.csv')
    ftab = ftab.loc[ftab['site'].map(lambda s: s in sites)]

    # total_sig = np.zeros(len(cnv_types))

    for sid in tqdm(set(ftab['sample_id'])):

        sample_fpath = os.path.join(src_path, sid + '.csv')
        dst_fpath = os.path.join(dst_path, sid + '.npy')

        # if os.path.isfile(dst_fpath):
        #     continue

        if os.path.isfile(sample_fpath):
            df = pd.read_csv(sample_fpath, dtype='str', usecols=['chrom_start', 'chrom_end'])
            cnv_sig = get_cnv_sig(df)
        else:
            cnv_sig = np.zeros(len(cnv_types))
        # total_sig += cnv_sig
        np.save(dst_fpath, cnv_sig)

    # print(total_sig / len(ftab))

if __name__ == "__main__":
    main()