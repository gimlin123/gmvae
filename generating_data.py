import random
import numpy as np
import pickle
import os
import os.path

# we will test with dim(z) = 50, dim(x) = 300
dim_z = 50
dim_x = 784
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


def generate_linear_dependence_low_var(dim_x, dim_y, dim_z, num_mixtures, train_points, test_points):
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

def generate_linear_dependence_low_label_var(dim_x, dim_y, dim_z, num_mixtures, train_points, test_points):
    means = np.random.rand(num_mixtures, dim_z)*5
    variances = np.random.rand(num_mixtures, dim_z)

    # random between -0.5 and 0.5
    mu_matrix = (np.random.rand(dim_x, dim_z)*10 - 5) / 10
    z_to_mu_lin = lambda z: np.dot(mu_matrix, z)

    #random between 0 and 0.1
    var_matrix = np.random.rand(dim_x, dim_z) / 10
    z_to_var_lin = lambda z: np.dot(var_matrix, z)

    # random between -0.5 and 0.5
    clus_mu_matrix = (np.random.rand(dim_y, dim_z)*10 - 5) / 10
    clus_to_mu_lin = lambda z: np.dot(clus_mu_matrix, z)

    #random between 0 and 0.0005
    clus_var_matrix = np.random.rand(dim_y, dim_z) / 2000
    clus_to_var_lin = lambda z: np.dot(clus_var_matrix, z)

    dataset = generate_dataset(means, variances, z_to_mu_lin, z_to_var_lin, clus_to_mu_lin, clus_to_var_lin, train_points, test_points)
    path = 'custom_data/low_label_variance.p'
    if os.path.isfile(path):
        os.remove(path)

    file = open(path, 'w')
    pickle.dump(dataset, file)

def generate_linear_dependence(dim_x, dim_y, dim_z, num_mixtures, train_points, test_points):
    means = np.random.rand(num_mixtures, dim_z)*5
    variances = np.random.rand(num_mixtures, dim_z)

    # random between -0.5 and 0.5
    mu_matrix = (np.random.rand(dim_x, dim_z)*10 - 5) / 10
    z_to_mu_lin = lambda z: np.dot(mu_matrix, z)

    #random between 0 and 0.1
    var_matrix = np.random.rand(dim_x, dim_z) / 10
    z_to_var_lin = lambda z: np.dot(var_matrix, z)

    # random between -0.5 and 0.5
    clus_mu_matrix = (np.random.rand(dim_y, dim_z)*10 - 5) / 10
    clus_to_mu_lin = lambda z: np.dot(clus_mu_matrix, z)

    #random between 0 and 0.1
    clus_var_matrix = np.random.rand(dim_y, dim_z) / 10
    clus_to_var_lin = lambda z: np.dot(clus_var_matrix, z)

    dataset = generate_dataset(means, variances, z_to_mu_lin, z_to_var_lin, clus_to_mu_lin, clus_to_var_lin, train_points, test_points)
    path = 'custom_data/linear.p'
    if os.path.isfile(path):
        os.remove(path)

    file = open(path, 'wb')
    pickle.dump(dataset, file)

def generate_polynomial_dependence(dim_x, dim_y, dim_z, num_mixtures, train_points, test_points, max_degree):
    means = np.random.rand(num_mixtures, dim_z)*5
    variances = np.random.rand(num_mixtures, dim_z)

    # random between -2 and 2
    z_to_mu_matrices = [np.random.rand(dim_x, dim_z)*4 - 2 for i in range(max_degree)]

    #random between 1 and 2
    z_to_var_matrices = [np.random.rand(dim_x, dim_z)*0.5 + 0.5 for i in range(max_degree)]

    # random between -2 and 2
    clus_to_mu_matrices = [np.random.rand(dim_y, dim_z)*4 - 2 for i in range(max_degree)]

    #random between 1 and 2
    clus_to_var_matrices = [np.random.rand(dim_y, dim_z)*0.5 + 0.5 for i in range(max_degree)]

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

    file = open(path, 'wb')
    pickle.dump(dataset, file)

def generate_mnist_cats():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    train_images_cats = np.random.randint(2, size=len(mnist.train.labels)).astype(np.bool)
    test_images_cats = np.random.randint(2, size=len(mnist.test.labels)).astype(np.bool)

    np.save('mnist_cats.npy', [train_images_cats, test_images_cats])

def generate_random_triplets(num_triplets):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    num_samples = 5000

    sample = np.random.choice(50000, num_samples, replace=False)
    images = mnist.train.images[sample]
    labels = mnist.train.labels[sample]

    label_dict = {}
    for i in range(len(labels)):
        label = labels[i]
        if label in label_dict:
            label_dict[label].append(i)
        else:
            label_dict[label] = [i]

    triplets = []
    for i in range(num_triplets):
        anchor = np.random.randint(num_samples)
        anchor_label = labels[anchor]
        anchor_group = label_dict[anchor_label]

        positive = anchor_group[np.random.randint(len(anchor_group))]

        neg_label = anchor_label
        negative = -1
        while neg_label == anchor_label:
            negative = np.random.randint(num_samples-1)
            if negative >= anchor:
                negative += 1
            neg_label = labels[negative]
        triplets.append([anchor, positive, negative])

    return images, triplets

def generate_random_triplets_custom(num_triplets, training_data):
    num_samples = 5000

    sample = np.random.choice(len(training_data['clusters']), num_samples, replace=False)
    images = training_data['data']
    labels = training_data['clusters']

    label_dict = {}
    for i in range(len(labels)):
        label = labels[i]
        if label in label_dict:
            label_dict[label].append(i)
        else:
            label_dict[label] = [i]

    triplets = []
    for i in range(num_triplets):
        anchor = np.random.randint(num_samples)
        anchor_label = labels[anchor]
        anchor_group = label_dict[anchor_label]

        positive = anchor_group[np.random.randint(len(anchor_group))]

        neg_label = anchor_label
        negative = -1
        while neg_label == anchor_label:
            negative = np.random.randint(num_samples-1)
            if negative >= anchor:
                negative += 1
            neg_label = labels[negative]
        triplets.append([anchor, positive, negative])

    return images, triplets

# order of similarity is:
    # same cat and samge label
    # different cat and same label
    # same cat and different label
    # different cat and different label
def generate_random_triplets_modified(num_triplets, train_images_cats):
    num_samples = 5000

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    sample = np.random.choice(50000, num_samples, replace=False)
    images = mnist.train.images[sample]
    labels = mnist.train.labels[sample]
    cats = train_images_cats[sample]
    indices = np.arange(num_samples)
    images[cats] *= config['mnist_scaling_factor']

    label_dict = {}
    for lab in range(2):
        label_dict[lab] = {}
        for i in range(10):
            label_dict[lab][i] = indices[np.all([labels == i, cats == lab], axis=0)]

    triplets = []
    for i in range(num_triplets):
        pos_cat = np.random.randint(3)
        neg_cat = np.random.randint(pos_cat, 3)

        anchor = np.random.randint(num_samples)
        anchor_label = labels[anchor]
        anchor_cat = cats[anchor]

        not_anchor_label_p = np.random.randint(9)
        if not_anchor_label_p >= anchor_label:
            not_anchor_label_p += 1

        pos_group = [label_dict[anchor_cat][anchor_label], label_dict[not(anchor_cat)][anchor_label],
            label_dict[anchor_cat][not_anchor_label_p]][pos_cat]
        positive = pos_group[np.random.randint(len(pos_group))]

        not_anchor_label_n = np.random.randint(9)
        if not_anchor_label_n >= anchor_label:
            not_anchor_label_n += 1

        neg_group = [label_dict[not(anchor_cat)][anchor_label], label_dict[anchor_cat][not_anchor_label_p],
            label_dict[not(anchor_cat)][not_anchor_label_p]][neg_cat]
        negative = neg_group[np.random.randint(len(neg_group))]
        triplets.append([anchor, positive, negative])

    return images, triplets

def format_triplets(images, triplets, batches):
    trip_arr = np.array(triplets)
    anchors = trip_arr[:, 0]
    positives = trip_arr[:, 1]
    negatives = trip_arr[:, 2]

    a_images_split = np.array(np.split(images[anchors], batches))
    p_images_split = np.array(np.split(images[positives], batches))
    n_images_split = np.array(np.split(images[negatives], batches))

    return np.concatenate((a_images_split, p_images_split, n_images_split), axis=1)

# generate_linear_dependence(dim_x, dim_y, dim_z, 10, 100000, 10000)
generate_polynomial_dependence(dim_x, dim_y, dim_z, 10, 100000, 10000, 5)

# generate_mnist_cats()
# generating triplets
#=======================================================
batches = 50
# images, triplets = generate_random_triplets(5000)
# formatted_triplets = format_triplets(images, triplets, batches)
# np.save(open("triplets.npy", "wb"), formatted_triplets)

# images, triplets = generate_random_triplets_modified(5000, train_images_cats)
# formatted_triplets = format_triplets(images, triplets, batches)
# np.save(open("triplets_modified.npy", "wb"), formatted_triplets)

custom_data = pickle.load(open("custom_data/polynomial.p", "rb" ))
images, triplets = generate_random_triplets_custom(5000, custom_data['train'])
formatted_triplets = format_triplets(images, triplets, batches)
np.save(open('triplets_custom.npy', 'wb'), formatted_triplets)
#=======================================================
