import configparser
config = configparser.ConfigParser()

config['gmvae_k'] = {
    'data': 'custom_data/polynomial.p',
    'data_type': 'real',
    'x_downscale': '1',
    'normalize_data': 'yes',

    'kl_loss_lambda' : '1',
    'reconstruct_loss_lambda': '1',

    'triplet_loss': 'no',
    'triplet_path': 'triplets_custom.npy',
    'tl_margin': '0.2',
    'tl_interleave_epoch': '250',
    'tl_lambda': '100',

    'scale_mnist': 'no',
    'mnist_scaling_factor': '0.5',
    'mnist_categories': 'mnist_cats.npy'
}

with open('gmvae.ini', 'w') as configfile:
    config.write(configfile)
