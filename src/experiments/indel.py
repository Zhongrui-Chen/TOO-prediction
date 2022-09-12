import pandas as pd

def main():
    df_indel = pd.read_csv('CosmicGenomeScreensMutantExport.tsv', encoding='ISO-8859-1', sep='\t')

if __name__ == '__main__':
    main()