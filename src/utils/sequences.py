import pysam

class RefNotFoundError(Exception):
    pass

class InvalidMutError(Exception):
    pass

def get_fasta_lookup_table(fasta_filepath):
    return pysam.FastaFile(fasta_filepath)

def get_chrom_idx_by_chrom(chrom):
    if chrom in ['X', 'Y']:
        chrom_idx= 23 if chrom == 'X' else 24
    else:
        chrom_idx = int(chrom)
    return chrom_idx

def get_chrom_by_idx(chrom_idx):
    if chrom_idx > 22:
        chrom = 'X' if chrom_idx == 23 else 'Y'
    else:
        chrom = str(chrom_idx)
    return chrom

def get_genomic_sequences(fasta_filepath):
    tb = get_fasta_lookup_table(fasta_filepath)
    chrom_refs = [ac for ac in tb.references if 'NC' in ac]
    gseqs = {}
    for chrom_idx in range(1, 25):
        chrom_ref = chrom_refs[chrom_idx - 1]
        chrom = get_chrom_by_idx(chrom_idx)
        gseqs[chrom] = tb.fetch(chrom_ref).upper()
    return gseqs

def nuc_at(seq, pos):
    return seq[int(pos) - 1]

def reverse_complement(nuc):
    if nuc == 'A':
        return 'G'
    elif nuc == 'T':
        return 'C'
    elif nuc == 'G':
        return 'A'
    elif nuc == 'C':
        return 'T'

def get_complement(nuc):
    mapping = {
        'G': 'C',
        'C': 'G',
        'A': 'T',
        'T': 'A'
    }
    return mapping[nuc]

def get_complementary_seq(dna_seq):
    compl = ''
    for nuc in dna_seq:
        compl += get_complement(nuc)
    return compl

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

# def sanity_check(gene, pos, ref):
def sanity_check(pos, ref, seq):
    pos = int(pos)
    if pos > len(seq) or ref != nuc_at(seq, pos):
        return False
    return True

def get_flanks(pos, seq):
    pos = int(pos)
    # if pos > 1:
    #     f5 = nuc_at(seq, pos-1)
    # else:
    #     f5 = '['
    # if pos < len(seq):
    #     f3 = nuc_at(seq, pos+1)
    # else:
    #     f3 = ']'
    f5 = nuc_at(seq, pos-1)
    f3 = nuc_at(seq, pos+1)
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