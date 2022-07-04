# def parse_hgvs(hgvs_str):

def is_range(pos):
    return '_' in str(pos)

# def parse_mut_type(hgvs_str):
#     ''' Input: var: hgvs.sequencevariant.SequenceVariant '''
#     # edit = str(var.posedit.edit)

#     if len(edit) == 3:
#         if edit[1] == '>':
#             mut_type = 'sub'
#         else:
#             # inv, del, dup
#             mut_type = edit
#     elif edit[:3] == 'ins':
#         mut_type = 'ins'
#     elif edit[:6] == 'delins':
#         mut_type = 'delins'
#     else:
#         mut_type = 'unk'
#     return mut_type

# def is_hgvs_valid(var):
#     ''' Check if a hgvs description is valid to our demand '''
#     pos = str(var.posedit.pos)
#     if '-' in pos or '*' in pos or '+' in pos:
#         return False
#     return True

def is_SNV(cmut):
    return '>' in cmut

def is_coding_mut(cmut):
    ''' Check if a hgvs description is a coding variant '''
    return not ('-' in cmut or '*' in cmut or '+' in cmut)

def parse_cmut(cmut):
    ''' e.g., ENST00000354590.7:c.317C>T '''
    seg = cmut.split('.')[-1] # 317C>T
    # Parse location
    for idx, ch in enumerate(seg):
        if ch.isalpha():
            sep_idx = idx
            break
    loc, mut = seg[:sep_idx], seg[sep_idx:]
    loc = int(loc)
    ref, alt = mut.split('>')
    return loc, ref, alt

def parse_chrom(gmut):
    ''' Parse the chromosome of a genomic HGVS description '''
    return gmut.split(':')[0]