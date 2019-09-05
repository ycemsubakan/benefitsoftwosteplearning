import numpy as np
import torch
from algorithms_v2 import VAE, audionet
import matplotlib.pyplot as plt
import utils as ut
import os
import visdom 
import pickle
import argparse
import librosa as lr

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev2')
assert vis.check_connection()

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=2)
arguments = argparser.parse_args()

arguments.cuda = torch.cuda.is_available()
arguments.batch_size = 200

np.random.seed(2)
torch.manual_seed(9)
arguments.cuda = torch.cuda.is_available()

arguments.data = 'synthetic_sounds' 

train_loader, wf = ut.preprocess_audio_files(arguments, overlap=True)

joint_tr = 1
results = []
EP = 200  # number of learning epochs
base_inits = 20

# now fit
Kdict = 370  # number of hidden states in the hmm p(r_t|r_{t-1})
Kss = [[80]]  # number of latent dimensions in the autoencoder p(h|r)
L = 800  # observation dimensionality p(x|h)

os.makedirs('models', exist_ok=True)

for config_num, Ks in enumerate(Kss):
    mdl = audionet(L, Ks[0], 1, 1, 1,
           base_inits=base_inits,
           Kdict=Kdict, 
           base_dist='HMM', 
           num_gpus=arguments.num_gpus, 
           usecuda=arguments.cuda,
           joint_tr=joint_tr)

    path = f'models/audionet_jointtr_{joint_tr}_{arguments.data}_K_{Ks}.t'

    if arguments.cuda:
        mdl.cuda()

    # Joint training ?
    tmp = mdl.base_dist
    mdl.base_dist = 'fixed_iso_gauss'
    mdl.VAE_trainer(arguments.cuda, train_loader, EP, vis = vis) 
    torch.save(mdl.state_dict(), path)

    mdl.base_dist = tmp        

    if not joint_tr:
        mdl.base_dist_trainer(train_loader, arguments.cuda, vis=vis, path=path)  
        if mdl.base_dist == 'GMM':
            pickle.dump(mdl.GMM, open(path + '.gmm', 'wb'))
        elif mdl.base_dist == 'HMM':
            pickle.dump(mdl.HMM, open(path + '.hmm', 'wb'))

    #av_lls, im_gen, im_test = compute_nparam_density(test_loader, NF, 0.2, arguments.cuda, num_samples=2)
    #results.append((av_lls, Ks))
    gen_data, seed = mdl.generate_data(1000, arguments)
    opts = {'title' : 'generated data {}'.format(model)}
    vis.line(gen_data.squeeze()[:3].t().data.cpu(), win='generated_{}'.format(model), opts=opts)

    gen_data_concat = ut.pt_to_audio_overlap(gen_data) 

    vis.line(gen_data_concat[:2000], win='generated_concat')
    vis.heatmap(seed.data.cpu()[:200].squeeze().t(), win='hhat-gen', opts={'title': 'Generated hhat'})
    
    lr.output.write_wav('sample_{}.wav'.format(arguments.data), gen_data_concat, 8000, norm=True)

# do some plotting here
imagepath = 'samples'
if not os.path.exists(imagepath):
    os.mkdir(imagepath)

Tstart = 120000
Tmax = 120000*2
pw = .3
plt.subplot(2, 1, 1)
spec = np.abs(lr.stft(wf[Tstart:Tmax]))**pw
lr.display.specshow(spec, sr=8000, y_axis='log', x_axis='time')
plt.title('Original Data')

plt.subplot(2, 1, 2)
spec = np.abs(lr.stft(gen_data_concat[Tstart:Tmax]))**pw
lr.display.specshow(spec, sr=8000, y_axis='log', x_axis='time')
plt.title('Generated Data')

plt.savefig(imagepath + '/{}.eps'.format(arguments.data), format='eps')
