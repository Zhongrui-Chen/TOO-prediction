def get_interested_sites():
    best_k = 10
    sites_of_interest = [
        # 'breast', 'kidney', 'liver', 'ovary', 'prostate', 'endometrium', 'large_intestine', 'lung', 'pancreas', 'skin' # 10 classes
        # 'ovary', 'lung', 'large_intestine', 'kidney', 'endometrium', 'breast' # 6 classes
        # 'haematopoietic_and_lymphoid_tissue', 'large_intestine', 'breast', 'lung', 'central_nervous_system', 
        # 'liver', 'kidney', 'prostate', 'pancreas', 'skin',
        # 'stomach', 'oesophagus', 'upper_aerodigestive_tract', 'thyroid', 'ovary', 'urinary_tract', 'biliary_tract'

        'skin', 'kidney', 'lung', 'central_nervous_system', 'large_intestine', 'stomach',
        'breast', 'pancreas',
        'cervix', 'thyroid',
        'urinary_tract', 'prostate', 'endometrium', 'liver', 'upper_aerodigestive_tract',
        'ovary', 'soft_tissue'
    ]
    return sites_of_interest[:best_k]