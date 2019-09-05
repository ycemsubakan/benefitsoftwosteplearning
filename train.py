import numpy as np
import torch
from algorithms_v2 import VAE, audionet
import utils as ut
import os
# import visdom 
import pickle
import argparse

# vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev2')
# assert vis.check_connection()
vis = None

# Parse command line args
argparser = argparse.ArgumentParser()
argparser.add_argument('mode', help='training mode', choices=['2step', 'joint', 'combined'])
argparser.add_argument('-s', '--seed', type=int, help='random seed', default=0)
argparser.add_argument('-g', '--gpu', type=int, help='GPU id', default=-1)

args = argparser.parse_args()

args.cuda = torch.cuda.is_available()
args.batch_size = 200
args.data = 'synthetic_sounds' 
args.num_gpus = 1

if args.gpu == -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'

EP = 200  # number of learning epochs
base_inits = 20  # for GMM init (not used)
Kdict = 100  # number of hidden states in the hmm p(c_t|c_{t-1})
Kss = [[80]]  # number of latent dimensions in the autoencoder p(h|r)
L = 800  # observation dimensionality p(x|h)

np.random.seed(args.seed)
torch.manual_seed(np.random.randint(np.iinfo(int).max))
args.cuda = torch.cuda.is_available()

train_loader, wf = ut.preprocess_audio_files(args, overlap=True)

for config_num, Ks in enumerate(Kss):
    mdl = audionet(L, Ks[0], 1, 1, 1, base_inits=base_inits, Kdict=Kdict, 
                   base_dist='HMM', 
                   num_gpus=args.num_gpus, 
                   usecuda=args.cuda,
                   joint_tr=1 if args.mode in ('combined', 'joint') else 0)

    if args.cuda:
        mdl.cuda()

    models_dir = f'models/K_{Ks}/{args.seed}/{args.mode}'
    os.makedirs(models_dir, exist_ok=False)

    path = f'{models_dir}/audionet.t'

    if args.mode == '2step':

        mdl.base_dist = 'fixed_iso_gauss'
        mdl.VAE_trainer(args.cuda, train_loader, EP, vis = vis) 
        torch.save(mdl.state_dict(), path)

        mdl.base_dist = 'HMM'        
        mdl.base_dist_trainer(train_loader, args.cuda, vis=vis, path=path)  

        pickle.dump(mdl.HMM, open(path + '.hmm', 'wb'))

    elif args.mode == 'joint':

        mdl.base_dist = 'fixed_iso_gauss'
        mdl.VAE_trainer(args.cuda, train_loader, EP, vis = vis) 
        mdl.base_dist = 'HMM'        

        torch.save(mdl.state_dict(), path)
        pickle.dump(mdl.HMM, open(path + '.hmm', 'wb'))

    elif args.mode == 'combined':
        path_init = f'models/K_{Ks}/{args.seed}/2step/audionet.t'

        assert os.path.exists(path_init)
        assert os.path.exists(path_init + '.hmm')

        mdl.load_state_dict(torch.load(path_init))
        mdl.HMM = pickle.load(open(path_init + '.hmm', 'rb'))

        # do extra training
        mdl.np_to_pt_HMM()
        mdl.base_dist = 'HMM'        
        mdl.VAE_trainer(args.cuda, train_loader, EP, vis = vis) 
        mdl.pt_to_np_HMM()
        torch.save(mdl.state_dict(), path)

        pickle.dump(mdl.HMM, open(path + '.hmm', 'wb'))

