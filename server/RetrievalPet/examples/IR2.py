import argparse
import os
import time
import pickle
import pdb
import csv
import numpy as np
import faiss
import torch
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from RetrievalPet.networks.imageretrievalnet import init_network, extract_vectors
from RetrievalPet.datasets.datahelpers import cid2filename
from RetrievalPet.datasets.testdataset_pet import configdataset
from RetrievalPet.utils.download import download_train, download_test
from RetrievalPet.utils.whiten import whitenlearn, whitenapply
from RetrievalPet.utils.evaluate import compute_map_and_print
from RetrievalPet.utils.general import get_data_root, htime
#
datasets_names = ['pet']
parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')
expansion = ['Spectral', '', 'AQE']
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='pet',
                    help="comma separated list of test datasets: " +
                         " | ".join(datasets_names) +
                         " (default: 'pet')")
def IR2(ranks):
    rank_len=10
    ## yuzhi, lianxushulinag while pick rank_len
    rank_re = np.loadtxt(os.path.join(result_dir, '{}_ranks_new_relevent.txt'.format(dataset)))

    ## the max value of rank_len
    MAX_RANK_LEN = int((rank_re.shape[0]) ** 0.5)
    rank_re=rank_re.reshape(MAX_RANK_LEN,MAX_RANK_LEN,rank_re.shape[1])
    for m in range(rank_re.shape[2]):
    for i in range(rank_re.shape[0]):
        rank_re[i][i][m]=1.0
    quanzhong=[1,0.7,0.4]+[0.1]*(MAX_RANK_LEN-3)
    for m in range(rank_re.shape[2]):
    #if adaption, then change the rank_len to a adaption params according to the rank_re_q, q_aer, cons_n
    if args.ir_adaption:
        using_local_query=True
        cons_n = 5
        q_aer = float(args.ir_adaption)
        if using_local_query:
            ## using local feature scores, please don't forget note the query_q belong to deep
            rank_re_q = np.loadtxt(os.path.join(result_dir, '{}_ranks_new_query.txt'.format(dataset)))
            query_q = rank_re_q[:, m]
        else:
            ## using deep feature scores
            query_q = scores[ranks[:, m], m]

        rank_len=0
        jishu=0
        for idx in range(min(len(query_q),MAX_RANK_LEN)-cons_n):
            if jishu<cons_n:
                if query_q[idx]>q_aer:
                    rank_len=idx+1
                else:
                    jishu+=1
            else:
                break
    max_dim = min(rank_len, MAX_RANK_LEN)
    print (max_dim)
    if max_dim>2:
        #put the image to the MAX_RANK_LEN2 location if equals max_dim then re rank in the maxdim length
        list2 = []
        list_hou = []
        MAX_RANK_LEN2 = max_dim
        for i in range(MAX_RANK_LEN2):
            if i < max_dim:
                fenshu = 0
                for j in range(max_dim):
                    fenshu+=rank_re[min(i,j)][max(i,j)][m]*quanzhong[j]
                fenshu = fenshu / (max_dim - 1)
                if fenshu > float(args.ir_remove):
                    list2.append(ranks[i][m])
                else:
                    list_hou.append(ranks[i][m])
            else:
                list2.append(ranks[i][m])
        ranks[0:MAX_RANK_LEN2, m] = list2 + list_hou
    return ranks

def main():
    args = parser.parse_args()

    # check if there are unknown datasets
#     pdb.set_trace()
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))
    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    vectore_dir='/root/server/best_model/se101_gem_cap/model_epoch1/'
    # extract database and query vectors

    cfg = configdataset("pet", get_data_root())
    print('>>  query images...')
    qvecs = np.load(os.path.join(vectore_dir, "pet_qvecs_ep1_resize_pca256_dba.npy"))
    print('>>  database images...')
    vecs = np.load(os.path.join(vectore_dir, "pet_vecs_ep1_resize_pca256.npy"))
    start = time.time()
    scores = np.dot(vecs, qvecs.T)
    ranks = np.argsort(-scores, axis=0)
    new_ranks=IR2(ranks)
    scores = np.dot(vecs, qvecs.T)
    compute_map_and_print(dataset, ranks, cfg['gnd_id'])
if __name__ == '__main__':
    main()