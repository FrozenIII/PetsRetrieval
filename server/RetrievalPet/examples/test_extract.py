import argparse
import os
import time
import pickle
import numpy as np

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

from RetrievalPet.networks.imageretrievalnet import init_network, extract_vectors
from RetrievalPet.datasets.datahelpers import cid2filename
from RetrievalPet.datasets.testdataset_pet import configdataset
from RetrievalPet.utils.whiten import whitenlearn, whitenapply
from RetrievalPet.utils.evaluate import compute_map_and_print
from RetrievalPet.utils.general import get_data_root, htime
from RetrievalPet.layers.functional import l2n

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}

datasets_names = ['pet']
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']
options = ['extract', 'evaluate']
parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')

# network
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                   help="network path, destination where network is saved")
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
                   help="network off-the-shelf, in the format 'ARCHITECTURE-POOLING' or 'ARCHITECTURE-POOLING-{reg-lwhiten-whiten}'," +
                        " examples: 'resnet101-gem' | 'resnet101-gem-reg' | 'resnet101-gem-whiten' | 'resnet101-gem-lwhiten' | 'resnet101-gem-reg-whiten'")

# test options

parser.add_argument('--datasets', '-d', metavar='DATASETS', default='pet',
                    help="comma separated list of test datasets: " +
                         " | ".join(datasets_names) +
                         " (default: 'kaggle_train_test')")
parser.add_argument('--image-size', '-imsize', default=326, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")

parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1]',
                    help="use multiscale vectors for testing, " +
                         " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--whitening', '-w', metavar='WHITENING', default=None, choices=whitening_names,
                    help="dataset used to learn whitening for testing: " +
                         " | ".join(whitening_names) +
                         " (default: None)")

# GPU ID
parser.add_argument('--gpu-id', '-g', default='2', metavar='N',
                    help="gpu id used for testing (default: '0')")
parser.add_argument('--using-cdvs',  default='0',
                    help='using cdvs global descriptor ')
parser.add_argument('--ir-remove',   default='0',
                    help='using ir remove action ')
parser.add_argument('--ir-adaption',   default='0',
                    help='adaption for ir-remove : example if ir_adaption equals to 0.3, the average map_n > 0.3 '
                         'and map_n+1 < 0.3, then we do the ir-remove in the n-rank list instead of a constant length list')
parser.add_argument('--use-caption', dest='use_caption', action='store_true',
                    help='run validation with caption')
def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    using_cdvs = float(args.using_cdvs)
    # loading network from path
    if args.network_path is not None:

        print(">> Loading network:\n>>>> '{}'".format(args.network_path))
        if args.network_path in PRETRAINED:
            # pretrained networks (downloaded automatically)
            state = load_url(PRETRAINED[args.network_path], model_dir=os.path.join(get_data_root(), 'networks'))
        else:
            # fine-tuned network from path
            state = torch.load(args.network_path)

        # parsing net params from meta
        # architecture, pooling, mean, std required
        # the rest has default values, in case that is doesnt exist
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get('local_whitening', False)
        net_params['regional'] = state['meta'].get('regional', False)
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False
        net_params['use_caption'] = args.use_caption
        # load network
        net = init_network(net_params)
        net.load_state_dict(state['state_dict'])

        # if whitening is precomputed
        if 'Lw' in state['meta']:
            net.meta['Lw'] = state['meta']['Lw']

        print(">>>> loaded network: ")
        print(net.meta_repr())

    # loading offtheshelf network
    elif args.network_offtheshelf is not None:

        # parse off-the-shelf parameters
        offtheshelf = args.network_offtheshelf.split('-')
        net_params = {}
        net_params['architecture'] = offtheshelf[0]
        net_params['pooling'] = offtheshelf[1]
        net_params['local_whitening'] = 'lwhiten' in offtheshelf[2:]
        net_params['regional'] = 'reg' in offtheshelf[2:]
        net_params['whitening'] = 'whiten' in offtheshelf[2:]
        net_params['pretrained'] = True

        # load off-the-shelf network
        print(">> Loading off-the-shelf network:\n>>>> '{}'".format(args.network_offtheshelf))
        net = init_network(net_params)
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))
    if len(ms) > 1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
        msp = net.pool.p.item()
        print(">> Set-up multiscale:")
        print(">>>> ms: {}".format(ms))
        print(">>>> msp: {}".format(msp))
    else:
        msp = 1

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # evaluate on test datasets
    datasets = args.datasets.split(',')

    result_dir=args.network_path[0:-8]+"/show_result"
    epoch_lun=args.network_path[0:-8].split('/')[-1].replace('model_epoch','')
    print(">> Creating directory if it does not exist:\n>> '{}'".format(result_dir))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for dataset in datasets:
        start = time.time()
        # search, rank, and print
        print('>> {}: Extracting...'.format(dataset))

        cfg = configdataset(dataset, get_data_root(),use_caption=args.use_caption)
        tuple_bbxs_qimlist = None
        tuple_bbxs_imlist = None
        images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
        qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
        # extract database and query vectors
        if args.use_caption:
            images_cap = [cfg['im_caption'](cfg, i) for i in range(cfg['n'])]
            qimages_cap = [cfg['qim_caption'](cfg, i) for i in range(cfg['nq'])]
        print('>> {}: query images...'.format(dataset))
        qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=tuple_bbxs_qimlist, ms=ms, msp=msp, batchsize=20)
        qvecs = qvecs.numpy()
        qvecs = qvecs.astype(np.float32)
        np.save(os.path.join(result_dir, "{}_qvecs_ep{}_resize.npy".format(dataset,epoch_lun)), qvecs)
#         qvecs = np.load(os.path.join(result_dir, "{}_qvecs_ep{}_resize.npy".format(dataset,epoch_lun)))
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, bbxs=tuple_bbxs_imlist, msp=msp, batchsize=20)
        vecs = vecs.numpy()
        vecs = vecs.astype(np.float32)
        np.save(os.path.join(result_dir, "{}_vecs_ep{}_resize.npy".format(dataset,epoch_lun)), vecs)
        if False:
            if using_cdvs!=0:
                print('>> {}: cdvs global descriptor loading...'.format(dataset))
                qvecs_global = cfg['qimlist_global']
                vecs_global = cfg['imlist_global']
                scores_global = np.dot(vecs_global, qvecs_global.T)
                scores+=scores_global*using_cdvs
            ranks = np.argsort(-scores, axis=0)
            if args.ir_remove!='0':
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
#             np.savetxt(os.path.join(result_dir, "{}_ranks.txt".format(dataset)), ranks.astype(np.int))
            np.savetxt(os.path.join('/home/donghuihui/data/CDVS_DATASET/paper_plot/', "{}_our_ranks.txt".format(dataset)), ranks.astype(np.int))
            compute_map_and_print(dataset, ranks, cfg['gnd_id'])


if __name__ == '__main__':
    main()