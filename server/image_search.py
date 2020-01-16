import os
import numpy as np
import faiss
import torch
from torchvision import transforms
from RetrievalPet.networks.imageretrievalnet import init_network, extract_vectors
from RetrievalPet.datasets.testdataset_pet import configdataset
from RetrievalPet.utils.general import get_data_root, htime
from ImageCaption import pets_cap
import random
import time
import sent2vec

class RetrievalModel:
    def __init__(self):
        self.queue=[]
        datasets=["pet_show_alldatabase"]
        dataset="pet_show_alldatabase"
        vectore_dir='EXPORT_DIR=petModel/pet_se_resnext101_32x4d_gem_contrastive_m0.70/_adam_lr1.0e-06_wd1.0e-04_gid1/_nnum5_qsize4000_psize2000/_bsize5_imsize1024/model_epoch1'
        self.cfg = configdataset(dataset, get_data_root())
        self.images = [self.cfg['im_fname'](self.cfg, i) for i in range(self.cfg['n'])]
        self.qimages = [self.cfg['qim_fname'](self.cfg, i) for i in range(self.cfg['nq'])]
        self.name_cluster = self.cfg['nameTolable']
        print('>>  query images...')
        qvecs = np.vstack([np.load(os.path.join(vectore_dir, "{}_qvecs_ep1_resize.npy".format(dataset))).astype('float32')for dataset in datasets])
        print('>>  database images...')
        self.vecs = np.vstack([np.load(os.path.join(vectore_dir, "{}_vecs_ep1_resize.npy".format(dataset))).astype('float32')for dataset in datasets])#可以换成DBA后的数据

        print(">> compute scores..")
        self.vecs = self.vecs.transpose(1, 0)#(1093759, 2048)  如果是DBA后的数据，这里不需要转置
        qvecs = qvecs.transpose(1, 0)#(115977, 2048)
        res = faiss.StandardGpuResources()
        dimension=self.vecs.shape[1]
        #     index_flat = faiss.index_factory(dimension,"PCA4096,PQ8 ",faiss.METRIC_INNER_PRODUCT)
        index_flat = faiss.IndexFlatIP(dimension)
        self.gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        self.gpu_index_flat.add(np.ascontiguousarray(self.vecs))
        k=100

        threshold = 0.09
        D, I = self.gpu_index_flat.search(np.ascontiguousarray(qvecs), 1)
        network_path = "best_model/se101_gem_cap/model_epoch1.pth.tar"

        multiscale="[1, 1/2**(1/2), 1/2]"
        state = torch.load(network_path)
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get('local_whitening', False)
        net_params['regional'] = state['meta'].get('regional', False)
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False
        net_params['use_caption'] = True
        self.net = init_network(net_params)
        self.net.load_state_dict(state['state_dict'])

        # if whitening is precomputed
        if 'Lw' in state['meta']:
            self.net.meta['Lw'] = state['meta']['Lw']

        print(">>>> loaded network: ")
        print(self.net.meta_repr())
        self.ms = list(eval(multiscale))
        if len(self.ms) > 1 and self.net.meta['pooling'] == 'gem' and not self.net.meta['regional'] and not self.net.meta['whitening']:
            self.msp = self.net.pool.p.item()
            print(">> Set-up multiscale:")
            print(">>>> ms: {}".format(self.ms))
            print(">>>> msp: {}".format(self.msp))
        else:
            self.msp = 1
        normalize = transforms.Normalize(
            mean=self.net.meta['mean'],
            std=self.net.meta['std']
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self.sequenceVevModel = sent2vec.Sent2vecModel()
        self.sequenceVevModel.load_model('sent2vec/wiki_unigrams.bin')

    def retrieval_image(self,image_path,caption,num_retrieval,isbase64):
        captionVector = self.sequenceVevModel.embed_sentences([caption])

        captionVector = captionVector

        k = int(num_retrieval)
        datapath=[image_path]
        qvecs = extract_vectors(self.net, datapath, 1024, self.transform, ms=self.ms, bbxs=None, msp=self.msp,
                                batchsize=1,isbase64=isbase64,noneedTcuda=False,use_cap=captionVector)
        qvecs = qvecs.numpy().transpose(1, 0)

        D, I = self.gpu_index_flat.search(np.ascontiguousarray(qvecs), k)
        Threshold = (np.sum(D[0][0:2]) / 2.0)
        Threshold2 = (np.sum(D[0][0:1]) / 1.0)
        Threshold_cut = 0.85
        print(Threshold,Threshold2)
        labelset=set()
        if Threshold2 < 0.999999:
            for ii in range(len(I[0])):
                if ii < 17:
                    labelset.add(self.name_cluster[I[0][ii]])
        if isbase64 and Threshold<0.85:
            return ([],1)
        index=0
        while(index<len(I[0])):
            if D[0][index]<Threshold_cut:
                break
            index+=1
        D = D[0][0:index]
        I = I[0][0:index]
        result_images=[self.images[i] for i in I]
        if isbase64 and Threshold2<0.999999:
            rannum = random.randint(0, 10)
            timenum = time.time()
            strrannum = str(rannum)
            strtimenum = str(timenum)
            rantime = "".join((strrannum, strtimenum))+".jpg"
            abpath=os.path.join(self.cfg['dir_images_q'],rantime)
            print(abpath)
            image_path.save(abpath)
            self.gpu_index_flat.add(qvecs)
            self.images.append(os.path.join(self.cfg['dir_images_q'],rantime))
            self.name_cluster.append(rantime)
            pets_cap[rantime]=caption
        if len(D)==0:
            return [],2
        if isbase64 and D[0]<0.95 and len(labelset)>2:
            return result_images,2
        if isbase64:
            self.queue.append(qvecs)
            return result_images, 0
        else:
            return result_images,0


    def getqimages(self):
        return self.qimages
    def getimages(self):
        return self.images
if __name__ == '__main__':
    model = RetrievalModel()
    model.retrieval_image(img_path)
#     img_path = r'D:\datasets\mnist\mnist_img\test\9\9.jpg'
#     result = model.predict(img_path, is_numpy=False,topk=3)
#     for label, prob in result:
#         print('label:%s,probability:%.4f'%(label, prob))