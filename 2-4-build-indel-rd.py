import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from math import floor
from lib.utils import get_genomic_bins, get_sites, normalized

def get_chrom_idx_by_chrom(chrom):
    if chrom in ['X', 'Y']:
        chrom_idx= 23 if chrom == 'X' else 24
    else:
        chrom_idx = int(chrom)
    return chrom_idx

def get_bin_idx(pos, chrom_idx, bins, bin_len=1000000):
    bin_idx = floor(int(pos) / bin_len) + (np.cumsum(bins)[chrom_idx-2] if chrom_idx > 1 else 0)
    return bin_idx

def get_indel_rd(df_ins, df_del):
    bins = get_genomic_bins()
    rd_ins = np.zeros(np.sum(bins))
    rd_del = np.zeros(np.sum(bins))

    for chrom, pos in zip(df_ins['chrom'], df_ins['chrom_start']):
        chrom = str(chrom)
        bin_idx = get_bin_idx(pos, get_chrom_idx_by_chrom(chrom), bins)
        rd_ins[bin_idx] += 1

    for chrom, pos in zip(df_del['chrom'], df_del['chrom_start']):
        chrom = str(chrom)
        bin_idx = get_bin_idx(pos, get_chrom_idx_by_chrom(chrom), bins)
        rd_del[bin_idx] += 1

    rd = rd_ins + rd_del

    # return normalized(rd_ins), normalized(rd_del)

    return normalized(rd)

# def get_indel_rd(df_ins, df_del):
#     bins = get_genomic_bins()
#     rd = np.zeros(np.sum(bins))

#     for chrom, pos in zip(df['chrom'], df['chrom_start']):
#         chrom = str(chrom)
#         bin_idx = get_bin_idx(pos, get_chrom_idx_by_chrom(chrom), bins)
#         rd[bin_idx] += 1

#     return normalized(rd)

def main():
    sites = get_sites('inter')
    src_path = './data/ICGC-dataset/simple'
    dst_path = './features/indel-rd'

   # Get the table
    ftab = pd.read_csv('./data/ICGC-dataset/sample-table.csv')
    ftab = ftab.loc[ftab['site'].map(lambda s: s in sites)]

    for sid in tqdm(set(ftab['sample_id'])):
        sample_fpath = os.path.join(src_path, sid + '.csv')
        dst_fpath = os.path.join(dst_path, sid + '.npy')

        # if os.path.isfile(dst_fpath):
        #     continue

        df = pd.read_csv(sample_fpath, dtype='str', usecols=['mut_type', 'chrom', 'chrom_start'])
        df = df[df['mut_type'].map(lambda t: 'insertion' in t or 'deletion' in t)]

        df_ins = df[df['mut_type'] == 'insertion of <=200bp']
        df_del  = df[df['mut_type'] == 'deletion of <=200bp']
        
        # rd_ins, rd_del = get_indel_rd(df_ins, df_del)
        rd = get_indel_rd(df_ins, df_del)
        # np.save(dst_fpath, np.concatenate([rd_ins, rd_del]))
        np.save(dst_fpath, rd)


if __name__ == "__main__":
    main()