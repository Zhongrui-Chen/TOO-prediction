import pandas as pd
import os
from lib.utils import get_sites

def preprocess_df(df, mapper):
    df = df.rename(mapper, axis=1)
    df = df[df['seq'] == 'WGS'] # Use WGS data
    df = df.drop_duplicates()
    # Check validity of chrom
    chrom_range = ['X', 'Y']
    for i in range(1, 22+1):
        chrom_range.append(str(i))
    chrom_criterion = df['chrom'].map(lambda c: c in chrom_range)
    df = df.loc[chrom_criterion]
    return df

def main():
    raw_simple_path = './data/ICGC-raw/simple'
    raw_cnv_path    = './data/ICGC-raw/cnv'

    dst_simple_path = './data/ICGC-processed/simple'
    dst_cnv_path    = './data/ICGC-processed/cnv'

    simple_mapper = {
        'icgc_sample_id': 'sample_id', 'project_code': 'proj_code', # sample
        'chromosome': 'chrom', 'chromosome_start': 'chrom_start', 'chromosome_end': 'chrom_end', # location
        'mutation_type': 'mut_type', 'mutated_from_allele': 'ref', 'mutated_to_allele': 'alt', # mutation type
        'sequencing_strategy': 'seq' # sequencing tech
    }

    cnv_mapper = {
        'icgc_sample_id': 'sample_id', 'project_code': 'proj_code', # sample
        'chromosome': 'chrom', 'chromosome_start': 'chrom_start', 'chromosome_end': 'chrom_end', # location
        'sequencing_strategy': 'seq' # sequencing tech
    }

    sites = get_sites('all')
    
    for s in sites:
        site_raw_simple_path = os.path.join(raw_simple_path, s)
        site_raw_cnv_path    = os.path.join(raw_cnv_path, s)
        site_dst_simple_path = os.path.join(dst_simple_path, s + '.csv')
        site_dst_cnv_path    = os.path.join(dst_cnv_path, s + '.csv')

        simple_proj_code_set = set()
        for f in os.listdir(site_raw_simple_path):
            # Ignore system files
            if f[0] == '.':
                continue
            proj_code = f.split('.')[-2]
            simple_proj_code_set.add(proj_code)

        cnv_proj_code_set = set()
        for f in os.listdir(site_raw_cnv_path):
            # Ignore system files
            if f[0] == '.':
                continue
            proj_code = f.split('.')[-2]
            cnv_proj_code_set.add(proj_code)
        
        proj_code_set = simple_proj_code_set.intersection(cnv_proj_code_set)

        sdf_list = []
        cdf_list = []
        for proj_code in proj_code_set:
            print(s, ':', proj_code, end=' => ')
            # Read simple file
            sf = os.path.join(site_raw_simple_path, 'simple_somatic_mutation.open.{}.tsv'.format(proj_code))
            cf = os.path.join(site_raw_cnv_path, 'copy_number_somatic_mutation.{}.tsv'.format(proj_code))
            simple_df = pd.read_csv(sf, sep='\t', usecols=simple_mapper.keys(), dtype='str') # specify dtype to avoid the warning of low memory
            cnv_df    = pd.read_csv(cf, sep='\t', usecols=cnv_mapper.keys(), dtype='str') # specify dtype to avoid the warning of low memory
            sdf = preprocess_df(simple_df, simple_mapper)
            cdf = preprocess_df(cnv_df, cnv_mapper)

            if len(sdf) == 0 or len(cdf) == 0:
                print('{} can be removed'.format(proj_code))
                continue
            else:
                sdf_list.append(sdf)
                cdf_list.append(cdf)
                print('processed')
        if len(sdf_list) == 0 and len(cdf_list) == 0:
            print('{} can be removed'.format(s))
        if len(sdf_list) > 0:
            pd.concat(sdf_list).to_csv(site_dst_simple_path, index=False)
        if len(cdf_list) > 0:
            pd.concat(cdf_list).to_csv(site_dst_cnv_path, index=False)

if __name__ == "__main__":
    main()