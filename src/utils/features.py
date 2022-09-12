from src.utils.sequences import get_complement
from itertools import product
from math import floor, ceil
import numpy as np
from src.utils.hgvs import is_range, parse_range
from src.utils.sequences import seq_between, get_chrom_by_idx

def get_bin_idx(pos, chrom_idx, bin_len, bins):
    pos = parse_range(pos)[0] if is_range(pos) else pos
    bin_idx = floor(int(pos) / bin_len) + (np.cumsum(bins)[chrom_idx-2] if chrom_idx > 1 else 0)
    return bin_idx

def get_contextualized_sbs_types():
    sub_types = [('C', x) for x in 'AGT'] + [('T', x) for x in 'ACG']
    contextualized_sbs_types = [x for x in product('ACGT', sub_types, 'ACGT')]
    return contextualized_sbs_types

def get_dbs_types():
    pass

def assign_sbs_type(ref, alt, f5, f3):
    if ref == 'G' or ref == 'A':
        return (get_complement(f5), (get_complement(ref), get_complement(alt)), get_complement(f3))
    else:
        return (f5, (ref, alt), f3)

def assign_tcn_category(tcn):
    # if tcn <= 2:
    #     return str(tcn)
    # elif tcn >= 3 and tcn <= 4:
    #     return '3-4'
    # elif tcn >= 5 and tcn <= 8:
    #     return '5-8'
    if tcn < 9:
        return str(tcn)
    else:
        return '9+'

def assign_size_category(size):
    million = 1000000
    if size > 0 and size <= 100000:
        return '0-100k'
    elif size > 100000 and size <= million:
        return '100k-1M'
    elif size > million and size <= 10 * million:
        return '1M-10M'
    elif size > 10 * million and size <= 40 * million:
        return '10M-40M'
    elif size > 40 * million:
        return '>40M'
    else:
        raise RuntimeError('Unknown size category: {}'.format(size))

# def assign_indel_type(mut_type, length, edit, range_bound):
#     if length == 1:
#         if edit == 'G' or edit == 'A':
#             edit = get_complement(edit)
#         return '{}1{}'.format(mut_type, edit)
#     elif length >= 2 and length < range_bound:
#         return '{}{}'.format(mut_type, length)
#     else:
#         return '{}{}+'.format(mut_type, str(range_bound))

def get_genomic_bins(genomic_seqs, num_chromosomes = 24, bin_len = 1000000):
    # Calculate the numbers of bins in each chromosome
    num_bins = np.zeros(num_chromosomes, dtype=int)
    for chrom_idx in range(1, num_chromosomes+1):
        chrom = get_chrom_by_idx(chrom_idx)
        num_bins[chrom_idx-1] = ceil(len(genomic_seqs[chrom]) / bin_len)
    return num_bins

def get_indel_types():
    indel_types = []
    for bp in range(1, 6):
        for mut_type in ['DEL', 'INS']:
            sublen_range = range(1, 7) if mut_type == 'DEL' else range(0, 6)
            for idx, sublen in enumerate(sublen_range):
                if bp == 1:
                    for edit in ['C', 'T']:
                        indel_type = []
                        bp_type = str(bp) + ('+' if bp == 5 else '') + 'bp'
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
                    bp_type = str(bp) + ('+' if bp == 5 else '') + 'bp'
                    indel_type.append(bp_type)
                    indel_type.append(mut_type)
                    indel_type.append('repeats')
                    if idx == len(sublen_range) - 1:
                        indel_type.append(str(sublen) + '+')
                    else:
                        indel_type.append(str(sublen))
                    indel_types.append('_'.join(indel_type))    
    return indel_types

def get_cnv_types():
    # TCN and size combinations

    # try: COSMIC CN signature

    cnv_types = []

    
    # for tcn in range(9):
    #     tcn_categories.append(str(tcn))
    # tcn_categories.append('9+')
    # het_states = ['HD', 'LOH', 'Het']

    # HD
    for size_ctg in ['0-100k', '100k-1M', '>1M']:
        cnv_types.append('_'.join(['HD', '0', size_ctg]))
    
    tcn_categories = ['1', '2', '3-4', '5-8', '9+']
    size_categories = ['0-100k', '100k-1M', '1M-10M', '10M-40M', '>40M']
    # LOH
    for tcn_ctg in tcn_categories:
        for size_ctg in size_categories:
            cnv_types.append('_'.join(['LOH', tcn_ctg, size_ctg]))
    # Het
    for tcn_ctg in tcn_categories[1:]:
        for size_ctg in size_categories:
            cnv_types.append('_'.join(['Het', tcn_ctg, size_ctg]))

    return cnv_types

def assign_het_state(tcn, minor):
    if tcn == 0:
        het_state = 'HD'
    elif minor == 0:
        het_state = 'LOH'
    elif minor > 0:
        het_state = 'Het'
    return het_state

def assign_tcn_category(tcn):
    # if tcn <= 2:
    #     return str(tcn)
    # elif tcn >= 3 and tcn <= 4:
    #     return '3-4'
    # elif tcn >= 5 and tcn <= 8:
    #     return '5-8'
    if tcn < 9:
        return str(tcn)
    else:
        return '9+'
    # if tcn < 3:
    #     return str(tcn)
    # elif tcn >= 3 and tcn <= 4:
    #     return '3-4'
    # elif tcn >= 5 and tcn <= 8:
    #     return '5-8'
    # elif tcn >= 9:
    #     return '9+'
    # else:
    #     raise RuntimeError('Unknown TCN category: {}'.format(tcn))

def assign_size_category(size, het_state):
    million = 1000000
    if size > 0 and size <= 100000:
        return '0-100k'
    elif size > 100000 and size <= million:
        return '100k-1M'
    else:
        # if het_state == 'HD':
        #     return '>1M'
        # else:
        if size > million and size <= 10 * million:
            return '1M-10M'
        elif size > 10 * million and size <= 40 * million:
            return '10M-40M'
        elif size > 40 * million:
            return '>40M'
        else:
            raise RuntimeError('Unknown size category: {}'.format(size))

def assign_cnv_type(tcn, size):
# def assign_cnv_type(tcn, size, minor):
    # het_state = assign_het_state(tcn, minor)

    return '_'.join([assign_tcn_category(tcn), assign_size_category(size)])
    # return '_'.join([het_state, assign_tcn_category(tcn), assign_size_category(size, het_state)])

def assign_indel_type(subtype, edit, seq, pos_start, pos_end):
    length = len(edit)
    indel_len_type = '5+bp' if length >= 5 else str(length) + 'bp'
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