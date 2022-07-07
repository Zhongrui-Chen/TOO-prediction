class ParseErrorException(Exception):
    pass

def is_range(pos):
    return '_' in str(pos)

def parse_mut_type(cmut):
    if '>' in cmut:
        return 'SNV'
    elif cmut in ['dup', 'del']:
        return 'CNV'
    elif cmut[:3] == 'ins' or cmut[:6] == 'delins':
        return 'INDEL'
    elif cmut == 'inv':
        return 'INV'
    else:
        raise ParseErrorException('Unknown mutation type: {}'.format(cmut))

def parse_cmut(hgvsc):
    seg = hgvsc.split('.')[-1]
    # Parse location
    for idx, ch in enumerate(seg):
        if ch.isalpha():
            sep_idx = idx
            break
    pos, mut = seg[:sep_idx], seg[sep_idx:]
    return pos, mut

def is_SNV(cmut):
    return '>' in cmut

def is_coding_mut(cmut):
    ''' Check if a hgvs description is a coding variant '''
    return not ('-' in cmut or '*' in cmut or '+' in cmut)

def parse_snv(mut):
    ''' C>T '''
    seg = mut.split('>')
    ref, alt = seg[0], seg[1]
    return ref, alt

def parse_indel(mut):
    if mut[:3] == 'ins':
        altseq = mut[3:]
    elif mut[:6] == 'delins':
        altseq = mut[6:]
    else:
        raise ParseErrorException('Unknown indel mutation: {}'.format(mut))
    return altseq

def parse_gmut(gmut):
    chrom = gmut.split(':')[0]
    seg = gmut.split('.')[-1] # 113087413G>T
    # Parse location
    for idx, ch in enumerate(seg):
        if ch.isalpha():
            sep_idx = idx
            break
    pos = seg[:sep_idx]
    return chrom, pos