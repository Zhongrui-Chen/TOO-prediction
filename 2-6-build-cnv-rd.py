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

bins = get_genomic_bins()

def get_cnv_rd(df):
    
    rd = np.zeros(np.sum(bins))

    for chrom, s, e in zip(df['chrom'], df['chrom_start'], df['chrom_end']):
        chrom = str(chrom)
        start_bin_idx = get_bin_idx(s, get_chrom_idx_by_chrom(chrom), bins)
        end_bin_idx   = get_bin_idx(e, get_chrom_idx_by_chrom(chrom), bins)
        n_bins = end_bin_idx - start_bin_idx + 1
        for idx in range(start_bin_idx, end_bin_idx):
            rd[idx] += 1 / n_bins

    return normalized(rd)

def main():
    sites = get_sites('inter')
    src_path = './data/ICGC-dataset/cnv'
    dst_path = './features/cnv-rd'

    # Get the table
    ftab = pd.read_csv('./data/ICGC-dataset/sample-table.csv')
    ftab = ftab.loc[ftab['site'].map(lambda s: s in sites)]

    for sid in tqdm(set(ftab['sample_id'])):
        sample_fpath = os.path.join(src_path, sid + '.csv')
        dst_fpath = os.path.join(dst_path, sid + '.npy')

        # if os.path.isfile(dst_fpath):
        #     continue
        
        if os.path.isfile(sample_fpath):
            df = pd.read_csv(sample_fpath, dtype='str', usecols=['chrom', 'chrom_start', 'chrom_end'])
            cnv_rd = get_cnv_rd(df)
            # total_n_cnvs += len(df['chrom'])
            # total_n += 1
        else:
            cnv_rd = np.zeros(np.sum(bins))

        np.save(dst_fpath, cnv_rd)

    # print(total_n_cnvs / total_n)


if __name__ == "__main__":
    main()