class ParseErrorException(Exception):
    pass

def is_range(pos):
    return '_' in str(pos)

def parse_range(pos):
    s, e = pos.split('_')
    return int(s), int(e)

def parse_indel_type(edit):
    if 'delins' in edit:
        return 'INDEL_DELINS'
    else:
        return 'INDEL_INS' if 'ins' in edit else 'INDEL_DEL'

def parse_mut_type(edit):
    if '>' in edit:
        return 'SBS'
    elif 'del' in edit or 'ins' in edit:
        return parse_indel_type(edit)
    elif 'dup' in edit:
        return 'DUP'
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
    mut_type = parse_mut_type(edit)
    return ac, ref_type, pos, edit, mut_type

# def is_SNV(cmut):
#     return '>' in cmut

def is_coding_mut(hgvsc):
    ''' Check if a HGVSC description is from a coding variant '''
    return not ('-' in hgvsc or '*' in hgvsc or '+' in hgvsc)

def parse_sbs_edit(edit):
    ''' C>T '''
    seg = edit.split('>')
    ref, alt = seg[0], seg[1]
    return ref, alt

def parse_ins_edit(edit):
    if 'delins' in edit:
        return edit.split('delins')[-1]
    elif 'ins' in edit:
        return edit.split('ins')[-1]

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