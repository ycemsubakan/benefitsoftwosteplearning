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
argparser.add_argument('-s', '--seed', type=int, help='random seed', default=0)
argparser.add_argument('mode', help='training mode', choices=['2step', 'joint', 'combined'])
args = argparser.parse_args()

args.cuda = torch.cuda.is_available()
args.batch_size = 200
args.data = 'synthetic_sounds' 
args.num_gpus = 1

np.random.seed(args.seed)
torch.manual_seed(np.random.randint(np.iinfo(int).max))
args.cuda = torch.cuda.is_available()

train_loader, wf = ut.preprocess_audio_files(args, overlap=True)

results = []
EP = 200
base_inits = 20

# now fit
Kdict = 370
L = 800

models_dir = f'models/{args.seed}'
os.makedirs(models_dir, exist_ok=True)

Kss = [[80]]
for config_num, Ks in enumerate(Kss):
    mdl = audionet(L, Ks[0], 1, 1, 1, base_inits=base_inits, Kdict=Kdict, 
                   base_dist='HMM', 
                   num_gpus=args.num_gpus, 
                   usecuda=args.cuda,
                   joint_tr=1 if args.mode in ('combined', 'joint') else 0)

    if args.cuda:
        mdl.cuda()

    path = f'{models_dir}/audionet_{args.mode}_K_{Ks}.t'

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
        torch.save(mdl.state_dict(), path)

        mdl.base_dist = 'HMM'        
        pickle.dump(mdl.HMM, open(path + '.hmm', 'wb'))

    elif args.mode == 'combined':
        assert os.path.exists(path)
        assert os.path.exists(path + '.hmm')

        mdl.load_state_dict(torch.load(path))
        mdl.HMM = pickle.load(open(path + '.hmm', 'rb'))

        # do extra training
        mdl.np_to_pt_HMM()
        mdl.VAE_trainer(args.cuda, train_loader, 1, vis = vis) 
        mdl.pt_to_np_HMM()

