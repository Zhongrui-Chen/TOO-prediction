import pandas as pd
from math import inf
from matplotlib import pyplot as plt
from collections import defaultdict, OrderedDict
from numpy import isnan
from tqdm import tqdm

def assign_tcn_category(tcn):
    # if tcn <= 2:
    #     return str(tcn)
    # elif tcn >= 3 and tcn <= 4:
    #     return '3-4'
    # elif tcn >= 5 and tcn <= 8:
    #     return '5-8'
    if tcn < 9:
        return str(tcn)
    else:
        return '9+'

def assign_size_category(size):
    million = 1000000
    if size > 0 and size <= 100000:
        return '0-100k'
    elif size > 100000 and size <= million:
        return '100k-1M'
    elif size > million and size <= 10 * million:
        return '1M-10M'
    elif size > 10 * million and size <= 40 * million:
        return '10M-40M'
    elif size > 40 * million:
        return '>40M'
    else:
        raise RuntimeError('Unknown size category: {}'.format(size))
    # else:
    #     # if het_state == 'HD':
    #     if tcn == 0:
    #         return '>1M'
    #     elif size > million and size <= 10 * million:
    #         return '1M-10M'
    #     elif size > 10 * million and size <= 40 * million:
    #         return '10M-40M'
    #     elif size > 40 * million:
    #         return '>40M'
    #     else:
    #         raise RuntimeError('Unknown size category!')

def main():
    print('Reading files...')
    ### df_mut = pd.read_csv('./data/processed/CosmicGenomeScreensMutantExport.tsv', sep='\t')
    df_cnv = pd.read_csv('./data/processed/CosmicCompleteCNA.tsv', sep='\t')
    
    print('Analysis...')
    label_dict = {}
    # tcn_cateogries = ['-2', '-1', '1', '2', '3-4', '5-8', '9+']
    tcn_categories = []
    for tcn in range(9):
        tcn_categories.append(str(tcn))
    tcn_categories.append('9+')
    size_categories = ['0-100k', '100k-1M', '1M-10M', '10M-40M', '>40M']

    # HD classes
    # for size_ctg in size_categories[:2] + ['>1M']:
    #     # label_dict[('HD', '0', size_ctg)] = 0
    #     label_dict['0', size_ctg] = 0
    
    # LOH & Het classes
    # for size_ctg in size_categories:
        # label_dict[('LOH', '1', size_ctg)] = 0

    # for size_ctg in size_categories:
    #     for tcn_ctg in tcn_cateogries:
    #         label_dict[('LOH', tcn_ctg, size_ctg)] = 0

    for size_ctg in size_categories:
        for tcn_ctg in tcn_categories:
            label_dict[(tcn_ctg, size_ctg)] = 0

    # tcn_set = set()

    tcn_list = df_cnv['total_cn']
    # minor_list = df_cnv['minor']
    coor_list = df_cnv['genomic_coordinates']

    # obs_dict = defaultdict(int)

    # for row in tqdm(df_cnv.itertuples(), total=len(df_cnv)):
    for idx in tqdm(range(len(df_cnv))):
        # minor = minor_list[idx]
        tcn = tcn_list[idx]

        # Drop CNVs with negative TCN
        if tcn < 0:
            continue

        # tcn_set.add(tcn)
        # minor = row.minor
        # minor_set.add(minor)
        coor = coor_list[idx]
        start, end = coor.split(':')[-1].split('..')
        start, end = int(start), int(end)
        size = end - start + 1
        # Classification
        # if tcn == 0:
            # het_state = 'HD'
        # elif tcn == 1:
            # het_state = 'LOH'
        # else:
            # if minor == 0:
            #     het_state = 'LOH'
            # elif minor > 0:
            #     het_state = 'Het'
        label_dict[(assign_tcn_category(tcn), assign_size_category(size))] += 1

        # if tcn in [3, 4]:
        #     if isnan(minor):
        #         obs_dict['NaN'] += 1
        #     else:
        #         obs_dict[minor] += 1

    # print('TCN set:', sorted(tcn_set))
    # print('minor set:', sorted(minor_set))
    
    print('There are {} classes'.format(len(label_dict.keys())))
    print(label_dict.keys())

    plt.figure(figsize=(10, 10))
    plt.bar_label(plt.barh(['_'.join(key) for key in label_dict.keys()], label_dict.values()))
    # od = OrderedDict(sorted(obs_dict.items()))
    # plt.bar_label(plt.barh(list(od.keys()), od.values()))
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == '__main__':
    main()