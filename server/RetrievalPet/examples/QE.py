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
parser.add_argument('--trained-network', '-tn', default='se50_gem/model_epoch,se101_gem/model_epoch,'+
                    'se101_spoc/model_epoch')
parser.add_argument('--gpu-id', '-g', default='1', metavar='N',
                    help="gpu id used for testing (default: '0')")
parser.add_argument('--query-expansion', '-qe', metavar='expansion', default='')
parser.add_argument('--linear-s', '-lis', type=int, default=30)
def main():
    args = parser.parse_args()

    # check if there are unknown datasets
#     pdb.set_trace()
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))
    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    vectore_dir='/root/server/best_model/se101_gem_cap/model_epoch/'
    # extract database and query vectors

    cfg = configdataset("pet", get_data_root())
    print('>>  query images...')
    qvecs = np.load(os.path.join(vectore_dir, "pet_qvecs_ep1_resize_pca256_dba.npy"))
    print('>>  database images...')
    vecs = np.load(os.path.join(vectore_dir, "pet_vecs_ep1_resize_pca256.npy"))
    start = time.time()
    scores = np.dot(vecs, qvecs.T)
    ranks = np.argsort(-scores, axis=0)
    compute_map_and_print(dataset, ranks, cfg['gnd_id'])


#
    print(">> compute scores..")
    res = faiss.StandardGpuResources()
    dimension = vecs.shape[1]

    #     index_flat = faiss.index_factory(dimension,"PCA4096,PQ8 ",faiss.METRIC_INNER_PRODUCT)
    index_flat = faiss.IndexFlatIP(dimension)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(np.ascontiguousarray(vecs))
    top_k = 20
    D, I = gpu_index_flat.search(np.ascontiguousarray(qvecs), top_k)  # actual search (115977, top_k)
    qe_qvecs = np.zeros((qvecs.shape), dtype=np.float32)
    for i in range(5):
        for na in range(qe_qvecs.shape[0]):
            qe_qvecs[na, :] = np.vstack((qvecs[na, :][np.newaxis, :], vecs[I[na, :top_k], :])).mean(0)
        if i != 4:
            D, I = gpu_index_flat.search(np.ascontiguousarray(qe_qvecs), top_k)

    # D, I = gpu_index_flat.search(np.ascontiguousarray(qe_qvecs), k)  # actual search
    scores = np.dot(vecs, qe_qvecs.T)
    ranks = np.argsort(-scores, axis=0)
    compute_map_and_print(dataset, ranks, cfg['gnd_id'])
if __name__ == '__main__':
    main()