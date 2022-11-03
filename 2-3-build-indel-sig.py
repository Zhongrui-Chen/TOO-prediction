from itertools import product
import pandas as pd
import numpy as np
import os
import pysam
from tqdm import tqdm

from lib.utils import get_complement, get_sites, nuc_at, normalized, get_genomic_sequences, seq_between

def get_indel_types():
    indel_types = []
    for bp in range(1, 8):
        for mut_type in ['DEL', 'INS']:
            sublen_range = range(1, 7) if mut_type == 'DEL' else range(0, 6)
            for idx, sublen in enumerate(sublen_range):
                if bp == 1:
                    for edit in ['C', 'T']:
                        indel_type = []
                        bp_type = str(bp) + ('+' if bp == 7 else '') + 'bp'
                        indel_type.append(bp_type)
                        indel_type.append(mut_type)
                        indel_type.append(edit)
                        indel_type.append('homo')
                        if idx == len(sublen_range) - 1:
                            indel_type.append(str(sublen) + '+')
                        else:
                            indel_type.append(str(sublen))
                        indel_types.append('_'.join(indel_type))
                else:
                    indel_type = []
                    bp_type = str(bp) + ('+' if bp == 7 else '') + 'bp'
                    indel_type.append(bp_type)
                    indel_type.append(mut_type)
                    indel_type.append('repeats')
                    if idx == len(sublen_range) - 1:
                        indel_type.append(str(sublen) + '+')
                    else:
                        indel_type.append(str(sublen))
                    indel_types.append('_'.join(indel_type))
    return indel_types

def assign_indel_type(subtype, edit, seq, pos_start, pos_end):
    length = len(edit)
    indel_len_type = '7+bp' if length >= 7 else str(length) + 'bp'
    # Create the window
    # window = edit
    sublen = 0 if subtype == 'INS' else 1
    # Extend the window leftwards
    for pos in range(pos_start - length, 1, -length):
        unit = seq_between(seq, pos, pos + length - 1)
        if unit != edit:
            break
        # window = unit + window
        sublen += 1
    # Extend the window rightwards
    for pos in range(pos_end + 1, len(seq) - length, length):
        unit = seq_between(seq, pos, pos + length - 1)
        if unit != edit:
            break
        # window = window + unit
        sublen += 1
    # Assign the INDEL sub-length type
    if subtype == 'DEL':
        sublen = '6+' if sublen >= 6 else str(sublen)
    if subtype == 'INS':
        sublen = '5+' if sublen >= 5 else str(sublen)
    # Determine the INDEL type
    if length == 1:
        single_base_indel_type = get_complement(edit) if edit in ['G', 'A'] else edit
        indel_type = '_'.join([indel_len_type, subtype, single_base_indel_type, 'homo' if length == 1 else 'repeats', sublen])
    else:
        indel_type = '_'.join([indel_len_type, subtype, 'homo' if length == 1 else 'repeats', sublen])
    # print(indel_type)
    return indel_type

genomic_fasta_path = './data/GCF_000001405.25_GRCh37.p13_genomic.fna'
genomic_seqs = get_genomic_sequences(genomic_fasta_path)
indel_types = get_indel_types()

def get_indel_sig(df_ins, df_del):
    sig = np.zeros(len(indel_types))

    for chrom, pos, alt in zip(df_ins['chrom'], df_ins['chrom_start'], df_ins['alt']):
        chrom = str(chrom)
        pos = int(pos)

        if 'N' in alt:
            continue

        indel_type = assign_indel_type('INS', alt, genomic_seqs[chrom], pos, pos)
        sig[indel_types.index(indel_type)] += 1
    
    for chrom, pos_start, pos_end, ref in zip(df_del['chrom'], df_del['chrom_start'], df_del['chrom_end'], df_del['ref']):
        chrom = str(chrom)
        pos_start = int(pos_start)
        pos_end   = int(pos_end)

        if 'N' in ref:
            continue

        indel_type = assign_indel_type('DEL', ref, genomic_seqs[chrom], pos_start, pos_end)
        sig[indel_types.index(indel_type)] += 1

    return normalized(sig)

def main():
    sites = get_sites('inter')
    src_path = './data/ICGC-dataset/simple'
    dst_path = './features/indel-sig'

    # Get the table
    ftab = pd.read_csv('./data/ICGC-dataset/sample-table.csv')
    ftab = ftab.loc[ftab['site'].map(lambda s: s in sites)]

    for sid in tqdm(set(ftab['sample_id'])):

        sample_fpath = os.path.join(src_path, sid + '.csv')
        dst_fpath = os.path.join(dst_path, sid + '.npy')

        # if os.path.isfile(dst_fpath):
        #     continue

        df = pd.read_csv(sample_fpath, dtype='str', usecols=['mut_type', 'chrom', 'chrom_start', 'chrom_end', 'ref', 'alt'])
        df = df[df['mut_type'].map(lambda t: 'insertion' in t or 'deletion' in t)]

        df_ins = df[df['mut_type'] == 'insertion of <=200bp']
        df_del  = df[df['mut_type'] == 'deletion of <=200bp']

        indel_sig = get_indel_sig(df_ins, df_del)

        # print(indel_sig)

        np.save(dst_fpath, indel_sig)

if __name__ == "__main__":
    main()