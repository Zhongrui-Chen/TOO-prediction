from click import edit
from src.utils.hgvs_parsing import parse_hgvs, is_range, parse_range, parse_ins_edit
from copy import copy

class MutAnnotation:
    def __init__(self, gene, hgvsg):
        if '_' in gene:
            gene = gene.split('_')[0]
        self.gene = gene
        self.chrom, self.ref_type, self.pos, self.edit, self.mut_type = parse_hgvs(hgvsg)

class SBS():
    def __init__(self, annotation):
        self.gene = annotation.gene
        ref, alt = annotation.edit.split('>')
        self.sub = (ref, alt)
    
    def __repr__(self) -> str:
        f5, f3 = self.flanks
        ref, alt = self.sub
        return 'SBS:{}:{}:{}[{}>{}]{}'.format(self.gene, self.bin, f5, ref, alt, f3)

    def set_flanks(self, f5, f3):
        self.flanks = (f5, f3)

    def set_bin(self, bin_idx):
        self.bin = bin_idx

class INDEL():
    def __init__(self, annotation):
        self.gene = annotation.gene
        self.type = annotation.mut_type
        if is_range(annotation.pos):
            pos_start, pos_end = parse_range(annotation.pos)
        else:
            pos_start, pos_end = annotation.pos, annotation.pos
        self.pos = (int(pos_start), int(pos_end))
        self.length = int(pos_end) - int(pos_start) + 1
        if self.type == 'INS':
            self.edit = parse_ins_edit(annotation.edit)
            self.length = len(self.edit)

    def __repr__(self) -> str:
        return '{}:{}:{}:{}:{}'.format(self.type, self.gene, self.bin, self.length, self.edit)

    def set_del_edit(self, edit):
        self.edit = edit

    def set_bin(self, bin_idx):
        self.bin = bin_idx

def split_delins(delins: MutAnnotation):
    ''' Split a DELINS annotation into a DEL and an INS '''
    del_annot = copy(delins)
    del_annot.mut_type = 'DEL'
    ins_annot = copy(delins)
    ins_annot.mut_type = 'INS'
    return (del_annot, ins_annot)