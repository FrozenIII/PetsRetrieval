import argparse
import os
import time
import numpy as np
import faiss
from RetrievalPet.datasets.testdataset_pet import configdataset
from RetrievalPet.utils.evaluate import compute_map_and_print
from RetrievalPet.utils.general import get_data_root, htime
import faiss
def main():
    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # evaluate on test datasets
    dataset = "pet"
    cfg = configdataset("pet_show_alldatabase", get_data_root())
    vectore_dir='best_model/se101_gem/model_epoch1/show_result'
    vecs=np.load(os.path.join(vectore_dir,"pet_show_alldatabase_vecs_ep1_resize.npy"))
    qvecs=np.load(os.path.join(vectore_dir, "pet_show_alldatabase_qvecs_ep1_resize.npy"))
    vecs= vecs.T
    qvecs = qvecs.T
    ori_dim = int(qvecs.shape[1])
    out_dim = 256

    # scores = np.dot(vecs, qvecs.T)
    # ranks = np.argsort(-scores, axis=0)
    # print(ranks.shape)
    # compute_map_and_print(dataset, ranks, cfg['gnd_id'])

    ##PCA method for test

    mat = faiss.PCAMatrix(ori_dim, out_dim)
    print(ori_dim,vecs.shape)
    mat.train(np.ascontiguousarray(vecs))
    assert mat.is_trained
    qvecs_pca = mat.apply_py(np.ascontiguousarray(qvecs))
    vecs_pca = mat.apply_py(np.ascontiguousarray(vecs))
    print(qvecs_pca.shape)

    np.save(os.path.join(vectore_dir, "pet_show_alldatabase_vecs_ep1_resize_pca.npy"),vecs_pca)
    np.save(os.path.join(vectore_dir, "pet_show_alldatabase_qvecs_ep1_resize_pca.npy"), qvecs_pca)
    # np.save(os.path.join(vectore_dir, "q_2{}.npy".format(out_dim)), qvecs_pca)

    # scores = np.dot(vecs_pca, qvecs_pca.T)
    # ranks = np.argsort(-scores, axis=0)
    # print(ranks.shape)
    # compute_map_and_print(dataset, ranks, cfg['gnd_id'])

if __name__ == '__main__':
    main()