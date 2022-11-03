from itertools import product
import pandas as pd
import numpy as np
import os
import pysam
from tqdm import tqdm

from lib.utils import get_complement, get_sites, nuc_at, normalized, get_genomic_sequences

def get_flanks(pos, seq):
    pos = int(pos)
    f5 = nuc_at(seq, pos-1)
    f3 = nuc_at(seq, pos+1)
    return f5, f3

def assign_sbs_type(ref, alt, f5, f3):
    if ref == 'G' or ref == 'A':
        return (get_complement(f5), (get_complement(ref), get_complement(alt)), get_complement(f3))
    else:
        return (f5, (ref, alt), f3)

genomic_fasta_path = './data/GCF_000001405.25_GRCh37.p13_genomic.fna'
genomic_seqs = get_genomic_sequences(genomic_fasta_path)
sub_types = [('C', x) for x in 'AGT'] + [('T', x) for x in 'ACG']
sbs_types = [x for x in product('ACGT', sub_types, 'ACGT')]

def get_sbs_sig(df):
    ''' return a 96-long vector '''    
    sig = np.zeros(len(sbs_types))

    for chrom, pos, ref, alt in zip(df['chrom'], df['chrom_start'], df['ref'], df['alt']):
        chrom = str(chrom)

        f5, f3 = get_flanks(pos, genomic_seqs[chrom])

        if ref == 'N' or alt == 'N' or f5 == 'N' or f3 == 'N':
            continue

        sbs_type = assign_sbs_type(ref, alt, f5, f3)
        if sbs_type in sbs_types:
            sig[sbs_types.index(sbs_type)] += 1

    return normalized(sig)

def main():
    sites = get_sites('inter')
    src_path = './data/ICGC-dataset/simple'
    dst_path = './features/sbs-sig'

    tab = pd.read_csv('./data/ICGC-dataset/sample-table.csv')
    tab = tab.loc[tab['site'].map(lambda x: x in sites)]

    # for s in sites:
    for r in tqdm(tab.itertuples(), total=len(tab)):
        f = os.path.join(src_path, r.sample_id + '.csv')
        df = pd.read_csv(f, dtype='str')

        dst_file_path = os.path.join(dst_path, r.sample_id + '.npy')

        # if os.path.isfile(dst_file_path):
        #     continue

        df = df[df['mut_type'] == 'single base substitution']

        # Build the SBS signature
        sbs_sig = get_sbs_sig(df)
        np.save(dst_file_path, sbs_sig)

if __name__ == "__main__":
    main()