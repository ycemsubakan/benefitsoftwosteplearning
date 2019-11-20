# Code for the paper ``On the Effectiveness of Two-Step Learning for Generative Models with Learnable Priors``. 

## Synthetic data experiment

The code for figure 4 is in the script `toydata_exps.py`.  

## MNIST GMM-VAE experiment 

The code for the traces in Figure 5 is is in `mnist_exps.py`. 


## Celeba GMM-VAE experiment

The code for the traces in Figure 7 is in `celeba_exps.py`.

## Audio experiment 

The code for the audio results is in `test_audio_all.py`

## Training a GAN prior for VAE

The code for this is in `wganpt/GAN_autoencoder_v2.py`. 

## Training a prior for GAN

The code this is in `wganpt/gan_train_forvae.py`. 

Generated images for a given VAE
!(VAE generated images)[wganpt/generations_vae_mnist.png]

Generated images after we learn a GAN on the latent space of the VAE
!(VAE generated images after learning a GAN on the latents)[wganpt/generations_vaeganprior_mnist.png]

