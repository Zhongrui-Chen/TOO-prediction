from src.utils.sequences import get_complement
from itertools import product
from math import floor
from numpy import cumsum
from src.utils.hgvs_parsing import is_range, parse_range

def get_bin_idx(pos, chrom_idx, bin_len, num_bins):
    pos = parse_range(pos)[0] if is_range(pos) else pos
    bin = floor(int(pos) / bin_len) + (cumsum(num_bins)[chrom_idx-2] if chrom_idx > 1 else 0)
    return bin

def get_contextualized_sbs_types():
    sub_types = [('C', x) for x in 'AGT'] + [('T', x) for x in 'ACG']
    # context_types = [x for x in product('[ACGT', sub_types, 'ACGT]')]
    contextualized_sbs_types = [x for x in product('ACGT', sub_types, 'ACGT')]
    return contextualized_sbs_types

def get_indel_types(range_bound):
    indel_types = []
    for mut_type in ['DEL', 'INS']:
        for edit in ['C', 'T']:
            indel_types.append(mut_type + '1' + edit)
        for length in range(2, range_bound):
            indel_types.append(mut_type + str(length))
        indel_types.append(mut_type + str(range_bound) + '+')
    return indel_types

def assign_sbs_type(ref, alt, f5, f3):
    if ref == 'G' or ref == 'A':
        return (f5, (get_complement(ref), get_complement(alt)), f3)
    else:
        return (f5, (ref, alt), f3)

def assign_indel_type(mut_type, length, edit, range_bound):
    if length == 1:
        if edit == 'G' or edit == 'A':
            edit = get_complement(edit)
        return '{}1{}'.format(mut_type, edit)
    elif length >= 2 and length < range_bound:
        return '{}{}'.format(mut_type, length)
    else:
        return '{}{}+'.format(mut_type, str(range_bound))