import argparse
import os

# Parse command line args
argparser = argparse.ArgumentParser()
argparser.add_argument('mode', help='training mode', choices=['2step', 'joint', 'combined'])
argparser.add_argument('k', help='number of latent dimensions in the autoencoder', type=int)
argparser.add_argument('measure', help='what to measure', choices=('nll', 'elbo', 'spectro'))
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
import itertools as it
import csv
import librosa as lr

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev2')
assert vis.check_connection()

args.cuda = torch.cuda.is_available()
args.batch_size = 200
args.data = 'synthetic_sounds' 
args.num_gpus = 1

K = 100  # number of latent dimensions in the autoencoder
Kdict = args.k  # number of hidden states in the hmm p(c_t|c_{t-1})
L = 800  # observation dimensionality p(x|h)

np.random.seed(args.seed)
torch.manual_seed(np.random.randint(np.iinfo(int).max))
args.cuda = torch.cuda.is_available()

model_dir = f'models/K_{args.k}/{args.seed}/{args.mode}'
vae_weights = f'{model_dir}/audionet.t'
hmm_weights = f'{model_dir}/audionet.t.hmm'

assert os.path.exists(vae_weights)
assert os.path.exists(hmm_weights)

train_data, test_data, wf = ut.preprocess_audio_files(args, overlap=True)
train_loader= torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
test_loader= torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

print(f"{len(train_data)} training samples")
print(f"{len(test_data)} test samples")

mdl = audionet(L, K, 1, 1, 1,
        Kdict=Kdict, 
        base_dist='HMM', 
        num_gpus=args.num_gpus, 
        usecuda=args.cuda)

if args.cuda:
    mdl.cuda()

mdl.load_state_dict(torch.load(vae_weights))
mdl.HMM = pickle.load(open(hmm_weights, 'rb'))

# make sure weights are sync
mdl.np_to_pt_HMM()
mdl.pt_to_np_HMM()


def risk_elbo(model, dataloader, cuda, n_samples=10, verbose=False):

    empirical_risk = 0
    for i, (dt, x, _) in enumerate(it.islice(dataloader, 0, None, 1)):
        if cuda:
            x = x.cuda()
        x = x.unsqueeze(1).float()

        with torch.no_grad():
            loss_dec, loss_enc, loss_prior = model.criterion_jointhmm(x, n_samples=n_samples, decouple=True)
            empirical_risk += (loss_enc + loss_dec + loss_prior).item()

        if verbose:
            print(f"losses: dec={loss_dec.item():.2f} KL={(loss_enc + loss_prior).item():.2f} ({loss_enc.item():.2f} {loss_prior.item():.2f})")

    empirical_risk /= len(dataloader)

    return empirical_risk


def risk_nll(model, dataloader, cuda, n_samples=5, verbose=False):

    empirical_risk = 0
    for i, (dt, x, _) in enumerate(it.islice(dataloader, 0, None, 1)):
        if cuda:
            x = x.cuda()
        x = x.unsqueeze(1).float()

        with torch.no_grad():
            loss = model.criterion_nll(x, n_samples=n_samples)
            empirical_risk += loss.item()

        if verbose:
            print(f"loss: nll={loss.item():.2f}")

    empirical_risk /= len(dataloader)

    return empirical_risk

if args.measure == 'elbo':

    print("Evaluating training set")
    train_elbo= risk_elbo(mdl, train_loader, args.cuda, verbose=True)

    print("Evaluating test set")
    test_elbo = risk_elbo(mdl, test_loader, args.cuda, verbose=True)

    print(f"NELBO: train={train_elbo:.2f} test={test_elbo:.2f}")

    with open(f'{model_dir}/nelbo.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['k', 'mode', 'seed', 'train_nelbo', 'test_nelbo'])
        writer.writeheader()
        writer.writerow({
            'k': args.k,
            'mode': args.mode,
            'seed': args.seed,
            'train_nelbo': train_elbo,
            'test_nelbo': test_elbo,
            })

elif args.measure == 'nll':

    print("Evaluating training set")
    train_nll = risk_nll(mdl, train_loader, args.cuda, verbose=True)

    print("Evaluating test set")
    test_nll = risk_nll(mdl, test_loader, args.cuda, verbose=True)

    print(f"NLL:   train={train_nll:.2f} test={test_nll:.2f}")

    with open(f'{model_dir}/nll.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['k', 'mode', 'seed', 'train_nll', 'test_nll'])
        writer.writeheader()
        writer.writerow({
            'k': args.k,
            'mode': args.mode,
            'seed': args.seed,
            'train_nll': train_nll,
            'test_nll': test_nll,
            })

elif args.measure == 'spectro':

    gen_data, seed = mdl.generate_data(1000, args)
    gen_data_concat = ut.pt_to_audio_overlap(gen_data)

    Tstart = 120000
    Tmax = 120000*2
    pw = .3
    # plt.subplot(2, 1, 1)
    spec = np.abs(lr.stft(wf[Tstart:Tmax]))**pw
    lr.display.specshow(spec, sr=8000, y_axis='log', x_axis='time')
    plt.title('Original Data')

    plt.savefig(f"{model_dir}/spectrogram_original.eps", format='eps')
    plt.savefig(f"{model_dir}/spectrogram_original.png", format='png')

    # plt.subplot(2, 1, 2)
    spec = np.abs(lr.stft(gen_data_concat[Tstart:Tmax]))**pw
    lr.display.specshow(spec, sr=8000, y_axis='log', x_axis='time')
    if args.mode == '2step':
        plt.title('Two-step training')
    elif args.mode == 'joint':
        plt.title('Joint training')
    elif args.mode == 'combined':
        plt.title('Combined training')

    plt.savefig(f"{model_dir}/spectrogram_{args.mode}.eps", format='eps')
    plt.savefig(f"{model_dir}/spectrogram_{args.mode}.png", format='png')

# gen_data, seed = mdl.generate_data(1000, args)
# gen_data_concat = ut.pt_to_audio_overlap(gen_data)
# librosa.output.write_wav(f'{model_dir}/sample.wav', gen_data_concat, 8000, norm=True)

# if vis is not None:
#     vis.line(gen_data.squeeze()[:3].t().data.cpu(),
#             win=f'generated_{model_dir}',
#             opts={'title' : f'generated data {model_dir}'})
#     vis.line(gen_data_concat[:2000],
#             win='generated_concat')
#     vis.heatmap(seed.data.cpu()[:200].squeeze().t(),
#             win='hhat-gen',
#             opts={'title': 'Generated hhat'})

