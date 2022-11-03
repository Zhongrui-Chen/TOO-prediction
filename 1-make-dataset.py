import os
from tqdm import tqdm
import pandas as pd
from lib.utils import get_sites

def main():
    simple_path     = './data/ICGC-processed/simple'
    cnv_path        = './data/ICGC-processed/cnv'
    dst_simple_path = './data/ICGC-dataset/simple'
    dst_cnv_path    = './data/ICGC-dataset/cnv'
    sites = get_sites('inter')

    simple_df_list = []
    cnv_df_list    = []
    for s in sites:
        simple_site_dict = {}
        cnv_site_dict    = {}
        site_simple_path = os.path.join(simple_path, s + '.csv')
        site_cnv_path    = os.path.join(cnv_path, s + '.csv')
        df_simple = pd.read_csv(site_simple_path, dtype='str')
        df_cnv    = pd.read_csv(site_cnv_path, dtype='str')

        for r in tqdm(df_simple.itertuples(), total=len(df_simple)):
            if r.sample_id in simple_site_dict:
                continue
            simple_site_dict[r.sample_id] = [s, r.proj_code]
            simple_sample_path = os.path.join(dst_simple_path, r.sample_id + '.csv')
            sipmle_sample_df = df_simple[df_simple['sample_id'] == r.sample_id]
            sipmle_sample_df.to_csv(simple_sample_path, index=False)
        
        for r in tqdm(df_cnv.itertuples(), total=len(df_cnv)):
            if r.sample_id in cnv_site_dict:
                continue
            cnv_site_dict[r.sample_id] = [s, r.proj_code]
            cnv_sample_path = os.path.join(dst_cnv_path, r.sample_id + '.csv')
            cnv_sample_df = df_cnv[df_cnv['sample_id'] == r.sample_id]
            cnv_sample_df.to_csv(cnv_sample_path, index=False)
        
        sdf = pd.DataFrame.from_dict(simple_site_dict, orient='index', columns=['site', 'proj_code'])
        sdf.index.name = 'sample_id'
        # cdf = pd.DataFrame.from_dict(cnv_site_dict, orient='index', columns=['site', 'proj_code'])
        # cdf.index.name = 'sample_id'
        simple_df_list.append(sdf)
        # cnv_df_list.append(cdf)

    df = pd.concat(simple_df_list)
    df.to_csv('./data/ICGC-dataset/sample-table.csv')
    # df = pd.concat(cnv_df_list)
    # df.to_csv('./data/ICGC-dataset/cnv-table.csv')

if __name__ == "__main__":
    main()