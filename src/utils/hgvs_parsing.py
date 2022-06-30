def is_range(pos):
    return '_' in str(pos)
    
def parse_mut_type(var):
    ''' Input: var: hgvs.sequencevariant.SequenceVariant '''
    edit = str(var.posedit.edit)
    if len(edit) == 3:
        if edit[1] == '>':
            mut_type = 'sub'
        else:
            # inv, del, dup
            mut_type = edit
    elif edit[:3] == 'ins':
        mut_type = 'ins'
    elif edit[:6] == 'delins':
        mut_type = 'delins'
    else:
        mut_type = 'unk'
    return mut_type

def is_hgvs_valid(var):
    ''' Check if a hgvs description is valid to our demand '''
    pos = str(var.posedit.pos)
    if '-' in pos or '*' in pos or '+' in pos:
        return False
    return True

def parse_chrom(var):
    ''' Parse the chromosome of a genomic HGVS description '''
    return var.ac.split('chr')[-1]