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

    def __init__(self, name, mode, imsize=None, nnum=5, qsize=2000, poolsize=20, transform=None, use_caption=False,loader=default_loader):

        if not (mode == 'train' or mode == 'val'):
            raise (RuntimeError("MODE should be either train or val, passed as string"))
        if name.startswith('pet'):
            # setting up paths
            data_root = get_data_root()
            db_root = data_root
            ims_root = os.path.join(data_root, 'images')

            # loading db
            db_fn = os.path.join(db_root, '{}.pkl'.format(name + "_train_hard"))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
            # setting fullpath for images
            self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        else:
            raise (RuntimeError("Unknown dataset name!"))
        with open("/root/server/images/pets_captions_vectors.pkl", 'rb') as f:
            self.caption_vectors = pickle.load(f)
        # initializing tuples dataset
        self.name = name
        self.mode = mode
        self.imsize = imsize
        self.clusters = db['cluster']
        self.cids = db['cids']
        self.qpool = db['qidxs']
        self.ppool = db['pidxs']
        self.nnum = nnum
        self.qsize = min(qsize, len(self.qpool))
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None
        self.use_caption=use_caption

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

        output = []
        # query image
        output.append(self.loader(self.images[self.qidxs[index]]))

        # positive image
        output.append(self.loader(self.images[self.pidxs[index]]))
        if self.use_caption:
            caption=[]
            caption.append(self.caption_vectors[self.cids[self.qidxs[index]]])
            caption.append(self.caption_vectors[self.cids[self.pidxs[index]]])
        # negative images
        for i in range(len(self.nidxs[index])):
            output.append(self.loader(self.images[self.nidxs[index][i]]))
            if self.use_caption:
                caption.append(self.caption_vectors[self.cids[self.nidxs[index][i]]])

        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]
        if self.transform is not None:
            output = [self.transform(output[i]).unsqueeze_(0) for i in range(len(output))]
        target = torch.Tensor([-1, 1] + [0] * len(self.nidxs[index]))
        if self.use_caption:
            output_new = []
            for i in range(len(output)):
                img = output[i]
                ca=caption[i]
                output_new.append((img,ca))
            return output_new,target
        else:
            return output, target

    def __len__(self):
        # if not self.qidxs:
        #     return 0
        # return len(self.qidxs)
        return self.qsize

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

    def create_epoch_tuples2(self, net):

        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------

        # draw qsize random queries for tuples
        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
        self.qidxs = [self.qpool[i] for i in idxs2qpool]
        self.pidxs = [self.ppool[i] for i in idxs2qpool]

        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------

        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return 0

        # draw poolsize random images for pool of negatives images
        ####if using less data
        newimages = []
        for i in range(len(self.images)):
            if self.clusters[i] != -1:
                newimages.append(i)
        random.shuffle(newimages)
        idxs2images = newimages[:self.poolsize]
        ### using all data
        # idxs2images = torch.randperm(len(self.images))[:self.poolsize]
        #test
        self.nidxs=[[]]*len(self.qidxs)
        for i in range(len(self.qidxs)):
            self.nidxs[i].append(self.qidxs[0])
    def create_epoch_tuples(self, net, temp_loss):

        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------

        # draw qsize random queries for tuples
        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
        self.qidxs = [self.qpool[i] for i in idxs2qpool]
        self.pidxs = [self.ppool[i] for i in idxs2qpool]

        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------

        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return 0

        # draw poolsize random images for pool of negatives images
        ####if using less data
        newimages = []
        for i in range(len(self.images)):
            if self.clusters[i] != -1:
                newimages.append(i)
        random.shuffle(newimages)
        idxs2images = newimages[:self.poolsize]
        # test
        #         self.nidxs=[[]]*len(self.qidxs)
        #         for i in range(self.nnum):
        #             self.nidxs[i].append(self.qidxs[0])
        ### using all data

        # idxs2images = torch.randperm(len(self.images))[:self.poolsize]

        # prepare network
        net.cuda()
        net.eval()
        batchsize = min(16, self.qsize)
        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():

            print('>> Extracting descriptors for query images...')
            # prepare query loader
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.images[i] for i in self.qidxs], imsize=self.imsize, re_size=True,
                               transform=self.transform),
                batch_size=batchsize, shuffle=False, num_workers=8, pin_memory=True
            )
            # extract query vectors
            qvecs = torch.zeros(net.meta['outputdim'], len(self.qidxs)).cuda()
            for i, input in enumerate(loader):
                if batchsize != 1:
                    if (i + 1) != len(loader):
                        qvecs[:, i * batchsize:(i + 1) * batchsize] = net(input.cuda(),None,False).data.squeeze()
                    else:
                        qvecs[:, i * batchsize:len(self.qidxs)] = net(input.cuda(),None,False).data.squeeze()
                else:
                    qvecs[:, i] = net(input.cuda(),None,False).data.squeeze()
                if (i + 1) % self.print_freq == 0 or (i + 1) == len(loader):
                    print('\r>>>> {}/{} done...'.format((i + 1), len(loader)), end='')
            print('')

            print('>> Extracting descriptors for negative pool...')
            # prepare negative pool data loader

            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.images[i] for i in idxs2images], imsize=self.imsize, re_size=True,
                               transform=self.transform),
                batch_size=batchsize, shuffle=False, num_workers=8, pin_memory=True
            )
            # extract negative pool vectors
            poolvecs = torch.zeros(net.meta['outputdim'], len(idxs2images)).cuda()
            for i, input in enumerate(loader):
                if batchsize != 1:
                    if (i + 1) != len(loader):
                        poolvecs[:, i * batchsize:(i + 1) * batchsize] = net(input.cuda(),None,False).data.squeeze()
                    else:
                        poolvecs[:, i * batchsize:len(idxs2images)] = net(input.cuda(),None,False).data.squeeze()
                else:
                    poolvecs[:, i] = net(input.cuda(),None,False).data.squeeze()
                if (i + 1) % self.print_freq == 0 or (i + 1) == len(loader):
                    print('\r>>>> {}/{} done...'.format((i + 1), len(loader)), end='')

            #                 poolvecs[:, i] = net(input.cuda(),None,False).data.squeeze()
            # if (i + 1) % self.print_freq == 0 or (i + 1) == len(idxs2images):
            #     print('\r>>>> {}/{} done...'.format(i + 1, len(idxs2images)), end='')
            print('')
            print('>> Searching for hard negatives...')
            torch.cuda.empty_cache()
            # compute dot product scores and ranks on GPU
            scores = torch.mm(poolvecs.t(), qvecs)

            scores, ranks = torch.sort(scores, dim=0, descending=True)
            avg_ndist = torch.tensor(0).float().cuda()  # for statistics
            n_ndist = torch.tensor(0).float().cuda()  # for statistics
            # selection of negative examples
            self.nidxs = []
            for q in range(len(self.qidxs)):
                # do not use query cluster,
                # those images are potentially positive
                qcluster = self.clusters[self.qidxs[q]]
                clusters = [qcluster]
                nidxs = []
                r = 0
                while len(nidxs) < self.nnum:
                    potential = idxs2images[ranks[r, q]]
                    # take at most one image from the same cluster
                    if (not self.clusters[potential] in clusters) and (self.clusters[potential] != -1):
                        nidxs.append(potential)
                        clusters.append(self.clusters[potential])
                        avg_ndist += torch.pow(qvecs[:, q] - poolvecs[:, ranks[r, q]] + 1e-6, 2).sum(dim=0).sqrt()
                        n_ndist += 1
                    r += 1
                self.nidxs.append(nidxs)
            print('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist / n_ndist))
            print('>>>> Done')

        return (avg_ndist / n_ndist).item()  # return average negative l2-distance
