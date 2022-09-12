import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt

def main():
    sites_filepath = './data/interim/dataset/sites.tsv'
    cnv_sample_id_list_filepath = './data/processed/CNVSampleIDList.tsv'
    cnv_sample_ids = pd.read_csv(cnv_sample_id_list_filepath, sep='\t')['sample_id'].to_list()
    sites_df = pd.read_csv(sites_filepath, sep='\t')
    sites_distro_dict = defaultdict(int)
    sample_ids = sites_df['sample_id'].to_list()
    sites = sites_df['site'].to_list()

    # for site in sites_df['site']:
    #     sites_distro_dict[site] += 1

    for idx in range(len(sites_df)):
        sid = sample_ids[idx]
        site = sites[idx]
        if sid in cnv_sample_ids:
            sites_distro_dict[site] += 1

    sites_distro_dict = dict(sorted(sites_distro_dict.items(), key=lambda item: item[1], reverse=True))

    # Display the 10 classes with most cases
    for k, _ in list(sites_distro_dict.items()):
        print('\'{}\''.format(k), end=', ')

    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    ax.bar_label(ax.barh(list(sites_distro_dict.keys()), sites_distro_dict.values()))
    ax.invert_yaxis()
    plt.show()

if __name__ == "__main__":
    main()