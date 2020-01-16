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
parser.add_argument('--trained-network', '-tn', default='se50_gem/model_epoch1,se101_gem/model_epoch1,'+
                    'se101_spoc/model_epoch3')
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
    expansion_m = args.query_expansion
    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    liner_s = args.linear_s
    # evaluate on test datasets
    datasets = args.datasets.split(',')
    vectore_dir='best_model/se101_gem/model_epoch1/show_result'
    # extract database and query vectors
    nets = args.trained_network.split(',')

    cfg = configdataset("pet", get_data_root())
    # print('>>  query images...')
    # qvecs = np.vstack([np.load(os.path.join(vectore_dir, i,
    #                                         "{}_qvecs_ep{}_resize.npy".format(
    #                                             dataset,i.split('/')[-1].replace('model_epoch','')))
    #                            ).astype('float32') for i in nets])
    # print('>>  database images...')
    # vecs = np.vstack([np.load(os.path.join(vectore_dir, i,
    #                                         "{}_vecs_ep{}_resize.npy".format(
    #                                             dataset, i.split('/')[-1].replace('model_epoch', '')))
    #                            ).astype('float32') for i in nets])
    # np.save(os.path.join(vectore_dir, "se50g_se101g_se101p_vecs_ep0_resize.npy"), vecs)
    # np.save(os.path.join(vectore_dir, "se50g_se101g_se101p_qvecs_ep0_resize.npy"), qvecs)

    print('>>  query images...')
    qvecs = np.load(os.path.join(vectore_dir, "pet_show_alldatabase_qvecs_ep1_resize_pca.npy"))
    print('>>  database images...')
    vecs = np.load(os.path.join(vectore_dir, "pet_show_alldatabase_vecs_ep1_resize_pca.npy"))
    start = time.time()
    # scores = np.dot(vecs, qvecs.T)
    # ranks = np.argsort(-scores, axis=0)
    # compute_map_and_print(dataset, ranks, cfg['gnd_id'])

    print(vecs.shape,qvecs.shape)
    print(">> compute scores..")
#     vecs = vecs.transpose(1, 0)#(1093759, 2048)
#     qvecs = qvecs.transpose(1, 0)#(115977, 2048)
    res = faiss.StandardGpuResources()
    dimension=vecs.shape[1]
    index_flat = faiss.IndexFlatIP(dimension)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

    gpu_index_flat.add(np.ascontiguousarray(vecs))
    top_k = 20
    D, I = gpu_index_flat.search(np.ascontiguousarray(qvecs), top_k)  # actual search (115977, top_k)
#
    qe_qvecs = np.zeros((qvecs.shape),dtype=np.float32)
    for na in tqdm(range(qe_qvecs.shape[0])):
        qe_qvecs[na,:]=np.vstack((qvecs[na,:][np.newaxis,:], vecs[I[na,:top_k], :])).mean(0)
    np.save(os.path.join(vectore_dir, "pet_show_alldatabase_qvecs_ep1_resize_dba.npy"),qe_qvecs)
    # scores = np.dot(vecs, qe_qvecs.T)
    # ranks = np.argsort(-scores, axis=0)
    # compute_map_and_print(dataset, ranks, cfg['gnd_id'])

    print('>> time: {}'.format(htime(time.time() - start)))

if __name__ == '__main__':
    main()