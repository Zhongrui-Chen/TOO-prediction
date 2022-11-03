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

def get_sbs_rd(df):
    bins = get_genomic_bins()
    rd = np.zeros(np.sum(bins))

    for chrom, pos, in zip(df['chrom'], df['chrom_start']):
        chrom = str(chrom)
        bin_idx = get_bin_idx(pos, get_chrom_idx_by_chrom(chrom), bins)
        rd[bin_idx] += 1

    return normalized(rd)

def main():
    sites = get_sites('inter')
    src_path = './data/ICGC-dataset/simple'
    dst_path = './features/sbs-rd'

    tab = pd.read_csv('./data/ICGC-dataset/sample-table.csv')
    tab = tab.loc[tab['site'].map(lambda x: x in sites)]

    for r in tqdm(tab.itertuples(), total=len(tab)):
        sid = r.sample_id

        sample_fpath = os.path.join(src_path, sid + '.csv')
        dst_fpath = os.path.join(dst_path, sid + '.npy')

        if os.path.isfile(dst_fpath):
            continue
        
        df = pd.read_csv(sample_fpath, dtype='str', usecols=['mut_type', 'chrom', 'chrom_start'])
        df = df[df['mut_type'] == 'single base substitution']

        sbs_rd = get_sbs_rd(df)
        np.save(dst_fpath, sbs_rd)

if __name__ == "__main__":
    main()