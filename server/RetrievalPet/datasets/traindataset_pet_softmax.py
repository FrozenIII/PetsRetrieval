import os
import pickle
import pdb
# import cv2
import torch
import torch.utils.data as data
import random
import numpy as np
from RetrievalPet.datasets.datahelpers import default_loader, imresize, cid2filename
from RetrievalPet.datasets.genericdataset import ImagesFromList
from RetrievalPet.utils.general import get_data_root
from PIL import Image

class TuplesDataset(data.Dataset):
    """Data loader that loads training and validation tuples of
        Radenovic etal ECCV16: CNN image retrieval learns from BoW

    Args:
        name (string): dataset name: 'retrieval-sfm-120k'
        mode (string): 'train' or 'val' for training and validation parts of dataset
        imsize (int, Default: None): Defines the maximum size of longer image side
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
        nnum (int, Default:5): Number of negatives for a query image in a training tuple
        qsize (int, Default:1000): Number of query images, ie number of (q,p,n1,...nN) tuples, to be processed in one epoch
        poolsize (int, Default:10000): Pool size for negative images re-mining

     Attributes:
        images (list): List of full filenames for each image
        clusters (list): List of clusterID per image
        qpool (list): List of all query image indexes
        ppool (list): List of positive image indexes, each corresponding to query at the same position in qpool

        qidxs (list): List of qsize query image indexes to be processed in an epoch
        pidxs (list): List of qsize positive image indexes, each corresponding to query at the same position in qidxs
        nidxs (list): List of qsize tuples of negative images
                        Each nidxs tuple contains nnum images corresponding to query image at the same position in qidxs

        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method,
            ie new q-p pairs are picked and negative images are remined
    """

    def __init__(self, name, mode, imsize=None, nnum=5, qsize=2000, poolsize=20, transform=None, sample_diy=False,loader=default_loader):

        if not (mode == 'train' or mode == 'val'):
            raise (RuntimeError("MODE should be either train or val, passed as string"))
        if name.startswith('pet'):
            # setting up paths
            data_root = get_data_root()
            db_root = data_root
            ims_root = os.path.join(data_root, 'images')

            # loading db
            db_fn = os.path.join(db_root, '{}.pkl'.format(name + "_train"))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
            # setting fullpath for images

            self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        else:
            raise (RuntimeError("Unknown dataset name!"))
        # initializing tuples dataset
        self.sample_diy=sample_diy
        self.name = name
        self.mode = mode
        self.imsize = imsize
        self.clusters = db['cluster']
        self.cluster_dic = db['cluster_num_dic']
        self.qpool = db['qidxs']
        self.ppool = db['pidxs']
        # size of training subset for an epoch
        self.nnum = nnum
        self.qsize = min(qsize, len(self.qpool))
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None

        self.transform = transform
        self.loader = loader

        self.print_freq = 10

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise (RuntimeError(
                "List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))
        output=self.loader(self.images[index])
        if self.imsize is not None:
            output = imresize(output, self.imsize)
        if self.transform is not None:
            output = self.transform(output).unsqueeze_(0)
        target = self.cluster_dic[self.clusters[index]]

        return output, target

    def __len__(self):
        # if not self.qidxs:
        #     return 0
        # return len(self.qidxs)
        return len(self.images)

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.qpool))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str