import os
import pickle
def configdataset(dataset, dir_main, use_caption=False):

    dataset = dataset.lower()
    gnd_fname = os.path.join(dir_main, '{}.pkl'.format(dataset+"_test"))
    print (gnd_fname," loaded")
    with open(gnd_fname, 'rb') as f:
        cfg = pickle.load(f)
    cfg['ext'] = ''
    cfg['qext'] = ''
    cfg['dir_data'] = dir_main
    cfg['dir_images_q'] = os.path.join(dir_main,'images')
    cfg['dir_images_index'] = os.path.join(dir_main,'images')
    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])
    cfg['qim_fname'] = config_qimname
    cfg['im_fname'] = config_imname
    cfg["caption_vectors"]=None
    cfg['qim_caption']=None
    cfg['im_caption']=None
    if use_caption:
        with open("/root/server/images/pets_captions_vectors.pkl", 'rb') as f:
            caption_vectors = pickle.load(f)
            cfg["caption_vectors"] = caption_vectors
            cfg['qim_caption'] = config_qcaption
            cfg['im_caption'] = config_icaption



    return cfg

def config_imname(cfg, i):
    return os.path.join(cfg['dir_images_index'], str(cfg['imlist'][i]) + cfg['ext'])

def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images_q'], str(cfg['qimlist'][i]) + cfg['qext'])

def config_icaption(cfg, i):
    return cfg["caption_vectors"][str(cfg['imlist'][i])]

def config_qcaption(cfg, i):
    return cfg["caption_vectors"][str(cfg['qimlist'][i])]


def config_imname2(cfg, i):
    img_name=str(cfg['imlist'][i])
    return os.path.join(cfg['dir_images_index'], img_name[0],img_name[1],img_name[2],img_name + cfg['ext'])

def config_qimname2(cfg, i):
    img_name=str(cfg['qimlist'][i])
    return os.path.join(cfg['dir_images_q'], img_name[0],img_name[1],img_name[2],img_name + cfg['qext'])

def config_imname_ori(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])

def config_qimname_ori(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])
