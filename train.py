import argparse
import os

# Parse command line args
argparser = argparse.ArgumentParser()
argparser.add_argument('mode', help='training mode', choices=['2step', 'joint', 'combined'])
argparser.add_argument('k', help='number of latent dimensions in the autoencoder', type=int)
argparser.add_argument('-s', '--seed', type=int, help='random seed', default=0)
argparser.add_argument('-g', '--gpu', type=int, help='GPU id', default=-1)

args = argparser.parse_args()

if args.gpu == -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'

import numpy as np
from algorithms_v2 import audionet
import utils as ut
# import visdom 
import pickle
import librosa
import torch

# vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev2')
# assert vis.check_connection()
vis = None

args.cuda = torch.cuda.is_available()
args.batch_size = 200
args.data = 'synthetic_sounds' 
args.num_gpus = 1

learning_rate = 1e-5  # learning rate
EP = 200  # number of learning epochs
base_inits = 20  # for GMM init (not used)
Kdict = 100  # number of hidden states in the hmm p(c_t|c_{t-1})
L = 800  # observation dimensionality p(x|h)

np.random.seed(args.seed)
torch.manual_seed(np.random.randint(np.iinfo(int).max))
args.cuda = torch.cuda.is_available()

train_loader, wf = ut.preprocess_audio_files(args, overlap=True)

mdl = audionet(L, args.k, 1, 1, 1, base_inits=base_inits, Kdict=Kdict, 
               base_dist='HMM', 
               num_gpus=args.num_gpus, 
               usecuda=args.cuda)

if args.cuda:
    mdl.cuda()

model_dir = f'models/K_{args.k}/{args.seed}/{args.mode}'
os.makedirs(model_dir, exist_ok=False)

path = f'{model_dir}/audionet.t'

if args.mode == '2step':

    mdl.VAE_trainer(args.cuda, train_loader, EP=EP, joint_training=False, vis=vis, lr=learning_rate) 

    mdl.base_dist_trainer(train_loader, args.cuda, vis=vis)  

elif args.mode == 'joint':

    # HMM initialization hack
    tmp = mdl.HMM.n_iter
    mdl.HMM.n_iter = 0
    mdl.base_dist_trainer(train_loader, args.cuda, vis=vis)  
    mdl.HMM.n_iter = tmp

    mdl.np_to_pt_HMM()
    mdl.VAE_trainer(args.cuda, train_loader, EP=EP, joint_training=True, vis=vis, lr=learning_rate) 
    mdl.pt_to_np_HMM()

elif args.mode == 'combined':

    path_init = f'models/K_{args.k}/{args.seed}/2step/audionet.t'
    assert os.path.exists(path_init)
    assert os.path.exists(path_init + '.hmm')

    mdl.load_state_dict(torch.load(path_init))
    mdl.HMM = pickle.load(open(path_init + '.hmm', 'rb'))

    # do extra training
    mdl.np_to_pt_HMM()
    mdl.VAE_trainer(args.cuda, train_loader, EP=EP, joint_training=True, vis=vis, lr=learning_rate) 
    mdl.pt_to_np_HMM()

torch.save(mdl.state_dict(), path)
pickle.dump(mdl.HMM, open(path + '.hmm', 'wb'))

gen_data, seed = mdl.generate_data(1000, args)

if vis is not None:
    vis.line(gen_data.squeeze()[:3].t().data.cpu(),
            win=f'generated_{model_dir}',
            opts={'title' : f'generated data {model_dir}'})
    vis.line(gen_data_concat[:2000],
            win='generated_concat')
    vis.heatmap(seed.data.cpu()[:200].squeeze().t(),
            win='hhat-gen',
            opts={'title': 'Generated hhat'})

gen_data_concat = ut.pt_to_audio_overlap(gen_data) 
librosa.output.write_wav(f'{model_dir}/sample.wav', gen_data_concat, 8000, norm=True)

