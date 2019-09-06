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
import torch
from algorithms_v2 import audionet
import matplotlib.pyplot as plt
import utils as ut
import visdom 
import pickle
import librosa

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev2')
assert vis.check_connection()

args.cuda = torch.cuda.is_available()
args.batch_size = 200
args.data = 'synthetic_sounds' 
args.num_gpus = 1

Kdict = 100  # number of hidden states in the hmm p(c_t|c_{t-1})
L = 800  # observation dimensionality p(x|h)

np.random.seed(args.seed)
torch.manual_seed(np.random.randint(np.iinfo(int).max))
args.cuda = torch.cuda.is_available()

train_loader, wf = ut.preprocess_audio_files(args, overlap=True)

mdl = audionet(L, args.k, 1, 1, 1,
        Kdict=Kdict, 
        base_dist='HMM', 
        num_gpus=args.num_gpus, 
        usecuda=args.cuda)

if args.cuda:
    mdl.cuda()

model_dir = f'models/K_{args.k}/{args.seed}/{args.mode}'
vae_weights = f'{model_dir}/audionet.t'
hmm_weights = f'{model_dir}/audionet.t.hmm'

assert os.path.exists(vae_weights)
assert os.path.exists(hmm_weights)

mdl.load_state_dict(torch.load(vae_weights))
mdl.HMM = pickle.load(open(hmm_weights, 'rb'))

gen_data, seed = mdl.generate_data(1000, args)
gen_data_concat = ut.pt_to_audio_overlap(gen_data) 
librosa.output.write_wav(f'{model_dir}/sample.wav', gen_data_concat, 8000, norm=True)

if vis is not None:
    vis.line(gen_data.squeeze()[:3].t().data.cpu(),
            win=f'generated_{model_dir}',
            opts={'title' : f'generated data {model_dir}'})
    vis.line(gen_data_concat[:2000],
            win='generated_concat')
    vis.heatmap(seed.data.cpu()[:200].squeeze().t(),
            win='hhat-gen',
            opts={'title': 'Generated hhat'})

# # do some plotting here
# imagepath = 'samples'
# if not os.path.exists(imagepath):
#     os.mkdir(imagepath)
# 
# Tstart = 120000
# Tmax = 120000*2
# pw = .3
# plt.subplot(2, 1, 1)
# spec = np.abs(lr.stft(wf[Tstart:Tmax]))**pw
# lr.display.specshow(spec, sr=8000, y_axis='log', x_axis='time')
# plt.title('Original Data')
# 
# plt.subplot(2, 1, 2)
# spec = np.abs(lr.stft(gen_data_concat[Tstart:Tmax]))**pw
# lr.display.specshow(spec, sr=8000, y_axis='log', x_axis='time')
# plt.title('Generated Data')
# 
# plt.savefig(imagepath + '/{}.eps'.format(args.data), format='eps')
