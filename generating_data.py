import random
import numpy as np
import pickle
import os
import os.path

# we will test with dim(z) = 50, dim(x) = 300
dim_z = 50
dim_x = 300
dim_y = 1
one_hot_dict = np.eye(dim_z)

def sample_gaussian_mixture(means, variances):
    num_mixtures = len(means)
    # which mixture to sample from
    ind = random.randint(0, num_mixtures-1)
    mu, var = (means[ind], variances[ind])
    return ind, np.random.normal(mu, var)

def generate_dataset(means, variances, z_to_mu, z_to_var, clus_to_mu, clus_to_var, train_points, test_points):
    dataset = {}
    dataset['train'] = {}
    dataset['train']['data'] = []
    dataset['train']['clusters'] = []
    dataset['train']['labels'] = []

    dataset['test'] = {}
    dataset['test']['data'] = []
    dataset['test']['clusters'] = []
    dataset['test']['labels'] = []

    for i in range(train_points):
        cluster, z = sample_gaussian_mixture(means,variances)
        dataset['train']['data'].append(np.random.normal(z_to_mu(z), z_to_var(z)))
        dataset['train']['clusters'].append(cluster)
        dataset['train']['labels'].append(np.random.normal(clus_to_mu(one_hot_dict[cluster]), clus_to_var(one_hot_dict[cluster])))

    for i in range(test_points):
        cluster, z = sample_gaussian_mixture(means,variances)
        dataset['test']['data'].append(np.random.normal(z_to_mu(z), z_to_var(z)))
        dataset['test']['clusters'].append(cluster)
        dataset['test']['labels'].append(np.random.normal(clus_to_mu(one_hot_dict[cluster]), clus_to_var(one_hot_dict[cluster])))

    dataset['train']['data'] = np.array(dataset['train']['data'])
    dataset['test']['data'] = np.array(dataset['test']['data'])
    dataset['train']['clusters'] = np.array(dataset['train']['clusters'])
    dataset['test']['clusters'] = np.array(dataset['test']['clusters'])
    dataset['train']['labels'] = np.array(dataset['train']['labels'])
    dataset['test']['labels'] = np.array(dataset['test']['labels'])
    return dataset


def generate_linear_dependence(dim_x, dim_z, num_mixtures, train_points, test_points):
    means = np.random.rand(num_mixtures, dim_z)*5
    variances = np.random.rand(num_mixtures, dim_z)

    # random between -0.5 and 0.5
    mu_matrix = (np.random.rand(dim_x, dim_z)*10 - 5) / 10
    z_to_mu_lin = lambda z: np.dot(mu_matrix, z)

    #random between 0 and 0.0005
    var_matrix = np.random.rand(dim_x, dim_z) / 2000
    z_to_var_lin = lambda z: np.dot(var_matrix, z)

    # random between -0.5 and 0.5
    clus_mu_matrix = (np.random.rand(dim_y, dim_z)*10 - 5) / 10
    clus_to_mu_lin = lambda z: np.dot(clus_mu_matrix, z)

    #random between 0 and 0.0005
    clus_var_matrix = np.random.rand(dim_y, dim_z) / 2000
    clus_to_var_lin = lambda z: np.dot(clus_var_matrix, z)

    dataset = generate_dataset(means, variances, z_to_mu_lin, z_to_var_lin, clus_to_mu_lin, clus_to_var_lin, train_points, test_points)
    path = 'custom_data/low_variance.p'
    if os.path.isfile(path):
        os.remove(path)

    file = open(path, 'w')
    pickle.dump(dataset, file)

def generate_polynomial_dependence(dim_x, dim_z, num_mixtures, train_points, test_points, max_degree):
    means = np.random.rand(num_mixtures, dim_z)*5
    variances = np.random.rand(num_mixtures, dim_z)

    # random between -5 and 5
    z_to_mu_matrices = [np.random.rand(dim_x, dim_z)*10 - 5 for i in range(max_degree)]

    #random between 0 and 0.333
    z_to_var_matrices = [np.random.rand(dim_x, dim_z) / 3 for i in range(max_degree)]

    # random between -5 and 5
    clus_to_mu_matrices = [np.random.rand(dim_y, dim_z)*10 - 5 for i in range(max_degree)]

    #random between 0 and 0.333
    clus_to_var_matrices = [np.random.rand(dim_y, dim_z) / 3 for i in range(max_degree)]

    def z_to_mu_poly(z):
        sum = np.zeros(dim_x)
        for i in range(max_degree):
            sum += np.dot(z_to_mu_matrices[i], z**i)
        return sum

    def z_to_var_poly(z):
        sum = np.zeros(dim_x)
        for i in range(max_degree):
            sum += np.dot(z_to_var_matrices[i], z**i)
        return sum

    def clus_to_mu_poly(z):
        sum = np.zeros(dim_x)
        for i in range(max_degree):
            sum += np.dot(clus_to_mu_matrices[i], z**i)
        return sum

    def clus_to_var_poly(z):
        sum = np.zeros(dim_x)
        for i in range(max_degree):
            sum += np.dot(clus_to_var_matrices[i], z**i)
        return sum

    dataset = generate_dataset(means, variances, z_to_mu_poly, z_to_var_poly, clus_to_mu_poly, clus_to_var_poly, train_points, test_points)
    path = 'custom_data/polynomial.p'
    if os.path.isfile(path):
        os.remove(path)

    file = open(path, 'w')
    pickle.dump(dataset, file)

generate_linear_dependence(dim_x, dim_z, 10, 100000, 10000)
# generate_polynomial_dependence(dim_x, dim_z, 10, 100000, 10000, 5)
