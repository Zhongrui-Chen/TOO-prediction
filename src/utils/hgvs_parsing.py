class ParseErrorException(Exception):
    pass

def is_range(pos):
    return '_' in str(pos)

def parse_range(pos):
    return pos.split('_')

def parse_mut_type(pos, edit):
    if '>' in edit:
        return 'SNV'
    elif 'del' in edit or 'dup' in edit or 'ins' in edit:
        if is_range(pos):
            rg_start, rg_end = parse_range(pos)
            if int(rg_end) - int(rg_start) + 1 >= 1000:
                return 'CNV'
        return 'INDEL'
    elif 'inv' in edit:
        return 'INV'
    else:
        raise ParseErrorException('Unknown mutation type: {}'.format(edit))

def parse_hgvs(hgvs):
    ac = hgvs.split(':')[0]
    ref_type = hgvs.split(':')[1][0]
    seg = hgvs.split('.')[-1]
    # Parse position or range
    for idx, ch in enumerate(seg):
        if ch.isalpha():
            sep_idx = idx
            break
    pos, edit = seg[:sep_idx], seg[sep_idx:]
    mut_type = parse_mut_type(pos, edit)
    return ac, ref_type, pos, edit, mut_type

# def is_SNV(cmut):
#     return '>' in cmut

def is_coding_mut(hgvsc):
    ''' Check if a HGVSC description is from a coding variant '''
    return not ('-' in hgvsc or '*' in hgvsc or '+' in hgvsc)

def parse_snv_edit(edit):
    ''' C>T '''
    seg = edit.split('>')
    ref, alt = seg[0], seg[1]
    return ref, alt

# def parse_indel(mut):
#     if mut[:3] == 'ins':
#         altseq = mut[3:]
#     elif mut[:6] == 'delins':
#         altseq = mut[6:]
#     else:
#         raise ParseErrorException('Unknown indel mutation: {}'.format(mut))
#     return altseq

# def parse_gmut(gmut):
#     chrom = gmut.split(':')[0]
#     seg = gmut.split('.')[-1] # 113087413G>T
#     # Parse location
#     for idx, ch in enumerate(seg):
#         if ch.isalpha():
#             sep_idx = idx
#             break
#     pos = seg[:sep_idx]
#     return chrom, pos