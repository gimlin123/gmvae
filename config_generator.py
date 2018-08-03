import configparser
config = configparser.ConfigParser()

config['gmvae_k'] = {
    'data_x': '784',
    'data': 'custom_data/polynomial.p',
#     'data': 'mnist',
    'data_type': 'real',
    'x_downscale': '1',
    'normalize_data': 'yes',
    'plot_data': 'yes',
    'x_min': '',
    'x_max': '',
    'var_downscale': '1',
    'conv_var_downscale': '1',
    'shuffle': 'yes',
    'save' : 'yes',
    
    'restore' : 'no',
    'restore_path' : '',
    
    'force_nondegeneracy': 'no',
    'batch_size': '32',
    'choice_ent_lambda': '1',
    
    'asin_path': '../../data/mike_data/cleaned_watch_data/asins/watch_asins_0.npy',
    'test_asin_path': '../../data/mike_data/cleaned_watch_data/asins/watch_asins_1.npy',
    'feature_path': '../../data/mike_data/cleaned_watch_data/features/watch_features_0.npy',
    'test_feature_path': '../../data/mike_data/cleaned_watch_data/features/watch_features_1.npy',
    'image_path': '../../data/mike_data/cleaned_watch_data/images/',
    'image_sample_cluster' : '50',

    'kl_loss_lambda' : '1',
    'reconstruct_loss_lambda': '1',
    'entropy_lambda' : '1',

    'triplet_loss': 'yes',
    'triplet_interleave': 'yes',
    'triplet_path': 'triplets_custom.npy',
    'tl_margin': '0.2',
    'tl_interleave_epoch': '250',
    'tl_lambda': '100',
    
    'tl_need_features': 'yes',
    'tl_asin_path': '../../data/mike_data/new_data/128p/asins.npy',
    'tl_feature_path': '../../data/mike_data/new_data/128p/features.npy',

    'scale_mnist': 'no',
    'mnist_scaling_factor': '0.5',
    'mnist_categories': 'mnist_cats.npy',
    
    'amazon_fashion_image': '../../data/mike_data/new_data/non_scraped_images/',
    'amazon_fashion_asin': '../../data/mike_data/new_data/128p/equal_cluster_data_15000/asins.npy',
    'amazon_fashion_feature': '../../data/mike_data/new_data/128p/equal_cluster_data_15000/features.npy',
    'amazon_fashion_label': '../../data/mike_data/new_data/128p/equal_cluster_data_15000/labels.npy',
    
    'embedding_size': '50',
    'layer_size': '512',
    'r_layer_size': '512',
    
    'tSNE' : 'yes'
}

config['linear_discriminator'] = {
    'data_x': '4096',
    'plot_data': 'yes',
    'min_margin': '3',
    
    'positive_features': '../../data/mike_data/cleaned_watch_data/positives/features.npy',
    'negative_features': '../../data/mike_data/cleaned_watch_data/negatives/features.npy',
    'asin_path': '../../data/mike_data/cleaned_watch_data/asins/',
    'feature_path': '../../data/mike_data/cleaned_watch_data/features/',
    'save_path': '../../data/mike_data/cleaned_watch_data/delete_asins.npy'
}

config['MDS'] = {
    'data': 'custom_data/polynomial.p',
    'plot_data' : 'yes',
    
    'n_components' : '2', 
    'n_init' : '4', 
    'max_iter' : '1000'
}

with open('gmvae.ini', 'w') as configfile:
    config.write(configfile)
