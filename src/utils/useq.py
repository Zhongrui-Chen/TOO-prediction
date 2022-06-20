def nuc_at(seq, loc):
    return seq[loc-1]

def seq_between(seq, start, end):
    # Included
    # e.g., ('ATGCA', 2, 4) => 'TGC'
    return seq[start-1 : end]

def get_references_by_gene(gene, lookup_table):
    '''
    Fuzzy lookup by gene
    In:  gene name, e.g., 'TP53'
    Out: Candidate list of references of possibly relevant sequences
    '''
    candidates = []
    for ref in lookup_table.references:
        if gene in ref:
            candidates.append(ref)
    if candidates:
        return candidates
    else:
        raise RuntimeError('No reference of the gene {} is found.'.format(gene))

def get_reference_by_gene_and_accession_number(gene, ac, lookup_table):
    '''
    Precise lookup by gene name and accession number
    In:  (gene name, accession number)
    Out: Candidate list of references of possibly relevant sequences
    '''
    refs = get_references_by_gene(gene, lookup_table)
    # Remove the version
    if '.' in ac:
        ac = ac.split('.')[0]
    candidates = []
    for ref in refs:
        if ac in ref:
            candidates.append(ref)
    if not candidates:
        candidates.append(gene)
    if len(candidates) == 1:
        return candidates[0]
    else:
        raise RuntimeError('Multiple candiate references of the gene {} are found.'.format(gene))