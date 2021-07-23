import pandas as pd
from collections import defaultdict
import scipy.sparse as sp
import logging
import numpy as np

""" Generate real dataset
"""
# paramter
val_test_size = 0.0001

# read file
data_path = 'data/'
combo_pd = pd.read_csv(data_path+'bio-decagon-combo.csv')
ppi_pd = pd.read_csv(data_path+'bio-decagon-ppi.csv')
tarAll_pd = pd.read_csv(data_path+'bio-decagon-targets-all.csv')


print("combo_pd:\n", combo_pd)
print("")
print("ppi_pd:\n", ppi_pd)
print("")
print("tarAll_pd:\n", tarAll_pd)
print("")


# build vocab


def build_vocab(words):
    vocab = defaultdict(int)
    for word in words:
        if word not in vocab.keys():
            vocab[word] = len(vocab)
    return vocab

# gene_list = list(ppi_pd['Gene 1'].unique()) + list(ppi_pd['Gene 2'].unique())
# drug_list = list(combo_pd['STITCH 1'].unique()) + list(combo_pd['STITCH 2'].unique())
# print(gene_list)


gene_list = list(ppi_pd['Gene 1'].unique()) + list(ppi_pd['Gene 2'].unique())
drug_list = list(combo_pd['STITCH 1'].unique()) + list(combo_pd['STITCH 2'].unique()) + list(tarAll_pd['STITCH'].unique())
gene_list_2 = list(tarAll_pd['Gene'].unique())
drug_list_2 = list(tarAll_pd['STITCH'].unique())

print(len(gene_list))
print(len(gene_list_2))

# gene_list = gene_list[:1000]
# drug_list = drug_list[:1000]
gene_vocab = build_vocab(gene_list)
drug_vocab = build_vocab(drug_list)

# print(gene_vocab)

# stat
n_genes = len(gene_vocab)
n_drugs = len(drug_vocab)
n_drugdrug_rel_types = len(combo_pd['Polypharmacy Side Effect'].unique())
print('# of gene %d' % n_genes)
print('# of drug %d' % n_drugs)
print('# of rel_types %d' % n_drugdrug_rel_types)

# TODO: # of gene unmatch

################# build gene-gene net #################
gene_stitch_list = set(tarAll_pd['Gene'].tolist())
print(f"gene_stich_list: {len(gene_stitch_list)}")
gene1_list, gene2_list = ppi_pd['Gene 1'].tolist(), ppi_pd['Gene 2'].tolist()
data_list, gene_idx1_list, gene_idx2_list = [], [], []
for u, v in zip(gene1_list, gene2_list):
    if u in gene_stitch_list and v in gene_stitch_list:
        u, v = gene_vocab.get(u, -1), gene_vocab.get(v, -1)
        # doesn't take any effects?
        if u == -1 or v == -1:
            continue
        data_list.extend([1, 1])
        gene_idx1_list.extend([u, v])
        gene_idx2_list.extend([v, u])
gene_adj = sp.csr_matrix((data_list, (gene_idx1_list, gene_idx2_list)))
print('gene-gene / protein-protein adj: {}\t{}\tnumber of edges: {}'.format(type(gene_adj), gene_adj.shape,
                                                                            gene_adj.nnz))
logging.info('{} --- {}'.format(gene_adj[u, v], gene_adj[v, u]))
gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()
print()

################# build gene-drug net #################
gene_ppi_list = set(list(ppi_pd['Gene 1'].unique()) + list(ppi_pd['Gene 2'].unique()))
print(f"gene_ppi_list: {len(gene_ppi_list)}")
stitch_list, gene_list = tarAll_pd['STITCH'].tolist(), tarAll_pd['Gene'].tolist()
# print(len(set(gene_list)))
# print(len(set(gene_ppi_list)))
data_list, drug_idx_list, gene_idx_list = [], [], []
for u, v in zip(stitch_list, gene_list):
    if v in gene_ppi_list:
        u, v = drug_vocab.get(u, -1), gene_vocab.get(v, -1)
        if u == -1 or v == -1:
            continue
        data_list.append(1)
        drug_idx_list.append(u)
        gene_idx_list.append(v)

# print(gene_idx_list)
# print(drug_idx_list)
# print(data_list)

gene_drug_adj = sp.csr_matrix((data_list, (gene_idx_list, drug_idx_list)))
drug_gene_adj = gene_drug_adj.transpose(copy=True)
print("gene_drug_adj" , gene_drug_adj.shape)

#logging.info('gene_drug_adj: {}'.format(gene_drug_adj.shape))
# logging.info('drug_gene_adj: {}'.format(drug_gene_adj.shape))
# tv, tu = 219, 5618
# logging.info('In gene-drug adj: {}'.format(gene_drug_adj[tu, tv]))
# logging.info('In drug-gene adj: {}'.format(drug_gene_adj[tv, tu]))
# print()

################# build drug-drug net #################
drug_drug_adj_list = []
drug1_list, drug2_list, se_list = combo_pd['STITCH 1'].tolist(), combo_pd['STITCH 2'].tolist(), combo_pd[
    'Polypharmacy Side Effect'].tolist()
se_dict = {}
for u, v, se in zip(drug1_list, drug2_list, se_list):
    u, v = drug_vocab.get(u, -1), drug_vocab.get(v, -1)
    if u == -1 or v == -1:
        continue
    if se not in se_dict:
        se_dict[se] = {'row': [], 'col': [], 'data': []}
    se_dict[se]['row'].extend([u, v])
    se_dict[se]['col'].extend([v, u])
    se_dict[se]['data'].extend([1, 1])

for key, value in se_dict.items():
    drug_drug_adj = sp.csr_matrix((value['data'], (value['row'], value['col'])), shape=(n_drugs, n_drugs))
    drug_drug_adj_list.append(drug_drug_adj)
    # print('Side Effect: {}'.format(key))
    # print('drug-drug network: {}\tedge number: {}'.format(drug_drug_adj.shape, drug_drug_adj.nnz))
logging.info('{} adjs with edges >= 500'.format(1098))

print("drug_drug_adj", drug_drug_adj.shape)
print("gene_adj", gene_adj.shape)

drug_drug_adj_list = sorted(drug_drug_adj_list, key=lambda x: x.nnz)[::-1][:964]
drug_drug_adj_list = drug_drug_adj_list[:10]
# drug_degree_list = map(lambda x: x.sum(axis=0).squeeze(), drug_drug_adj_list)
print('# of filtered rel_types:%d' % len(drug_drug_adj_list))
drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]
for i in range(10):
    logging.info('shape:{}\t{} match {}'.format(drug_drug_adj_list[i].shape, drug_drug_adj_list[i].nnz,
                                                np.sum(drug_degrees_list[i])))
print()
print('Done data loading')


print("=" * 40)
print("THIS!")
print(drug_drug_adj_list)
print(len(drug_degrees_list))
print("=" * 40)
