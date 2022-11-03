import pandas as pd
from sklearn.model_selection import train_test_split

from lib.utils import get_sites

def main():
    sites = get_sites('f')

    tab = pd.read_csv('./data/ICGC-dataset/sample-table.csv')
    tab = tab.loc[tab['site'].map(lambda s: s in sites)]

    samples = []
    labels = []
    for r in tab.itertuples():
        samples.append(r.sample_id)
        labels.append(sites.index(r.site))

    id_train, id_test = train_test_split(samples, test_size=0.2, shuffle=True, stratify=labels)

    df_train = tab.loc[tab['sample_id'].map(lambda x: x in id_train)]
    df_train.to_csv('./data/ICGC-dataset/f-train-table.csv', index=False)

    df_test  = tab.loc[tab['sample_id'].map(lambda x: x in id_test)]
    df_test.to_csv('./data/ICGC-dataset/f-test-table.csv', index=False)

if __name__ == "__main__":
    main()