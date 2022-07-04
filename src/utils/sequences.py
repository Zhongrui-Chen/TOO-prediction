import pysam

class RefNotFoundError(Exception):
    pass

class InvalidMutError(Exception):
    pass

cds_fasta_filepath = './data/external/All_COSMIC_Genes.fasta'
def get_cds_lookup_table():
    return pysam.FastaFile(cds_fasta_filepath)

def nuc_at(seq, loc):
    return seq[loc-1]

def reverse_complement(nuc):
    if nuc == 'A':
        return 'G'
    elif nuc == 'T':
        return 'C'
    elif nuc == 'G':
        return 'A'
    elif nuc == 'C':
        return 'T'

def assign_mut_type(ref, alt, f5, f3):
    mapping = {
        'G': 'C',
        'C': 'G',
        'A': 'T',
        'T': 'A'
    }
    if ref == 'G' or ref == 'A':
        return (f5, (mapping[ref], mapping[alt]), f3)
    else:
        return (f5, (ref, alt), f3)

def seq_between(seq, start, end):
    # Included, e.g., ('ATGCA', 2, 4) => 'TGC'
    return seq[start-1 : end]

def get_reference(gene, lookup_table):
    if gene in lookup_table.references:
        return gene
    else:
        raise RefNotFoundError('Reference of {} is not found'.format(gene))

def get_sequence(gene, lookup_table):
    ref = get_reference(gene, lookup_table)
    return lookup_table.fetch(ref).upper()

def get_flanks(gene, loc, ref, lookup_table):
    seq = get_sequence(gene, lookup_table)
    # Sanity check
    # if loc > len(seq):
    #     raise InvalidMutError('Loc out of range: loc={}, seqlen={}'.format(loc, len(seq)))
    # elif ref != nuc_at(seq, loc):
    #     raise InvalidMutError('Ref not matched: ref={}, seq[loc]={}'.format(ref, nuc_at(seq, loc)))
    if loc > len(seq) or ref != nuc_at(seq, loc):
        return None
    if loc > 1:
        f5 = nuc_at(seq, loc-1)
    else:
        f5 = '<'
    if loc < len(seq):
        f3 = nuc_at(seq, loc+1)
    else:
        f3 = '>'
    return f5, f3

# def get_references_by_gene(gene, lookup_table):
#     '''
#     Fuzzy lookup by gene
#     In:  gene name, e.g., 'TP53'
#     Out: Candidate list of references of possibly relevant sequences
#     '''
#     candidates = []
#     for ref in lookup_table.references:
#         if gene in ref:
#             candidates.append(ref)
#     if candidates:
#         return candidates
#     else:
#         raise RuntimeError('No reference of the gene {} is found.'.format(gene))

# def get_reference_by_gene_and_accession_number(gene, ac, lookup_table):
#     '''
#     Precise lookup by gene name and accession number
#     In:  (gene name, accession number)
#     Out: Candidate list of references of possibly relevant sequences
#     '''
#     refs = get_references_by_gene(gene, lookup_table)
#     # Remove the version
#     if '.' in ac:
#         ac = ac.split('.')[0]
#     candidates = []
#     for ref in refs:
#         if ac in ref:
#             candidates.append(ref)
#     if not candidates:
#         candidates.append(gene)
#     if len(candidates) == 1:
#         return candidates[0]
#     else:
#         raise RuntimeError('Multiple candiate references of the gene {} are found.'.format(gene))