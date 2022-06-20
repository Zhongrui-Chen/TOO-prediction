import pysam
cds_fasta_filepath = './data/external/All_COSMIC_Genes.fasta'
def get_cds_lookup_table():
    return pysam.FastaFile(cds_fasta_filepath)