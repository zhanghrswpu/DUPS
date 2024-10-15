import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

cross_entropy_generator_and_discriminator = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy_generator_and_discriminator(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy_generator_and_discriminator(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy_generator_and_discriminator(tf.ones_like(fake_output), fake_output)


def get_parameter(dataset_name, non_iid_p, malicious_clients_p, aggregation_method, defense_method,
                  attack_mothod):
    dataset_and_model_mnist = {
        'dataset_name': 'MNIST',
        'model_name': 'CNN',
        'input_shape': [None, 28, 28, 1],
        'classes_num': 10,
        'target_label': 3,
        'num_client': 100
    }
    dataset_and_model_fmnist = {
        'dataset_name': 'FMNIST',
        'model_name': 'CNN',
        'input_shape': [None, 28, 28, 1],
        'classes_num': 10,
        'target_label': 8,
        'num_client': 100
    }
    dataset_and_model_chmnist = {
        'dataset_name': 'CH-MNIST',
        'model_name': 'RES',
        'input_shape': [None, 128, 128, 3],
        'classes_num': 8,
        'num_client': 40
    }
    dataset_and_model_cifar10 = {
        'dataset_name': 'cifar-10',
        'model_name': 'RES',
        'input_shape': [None, 32, 32, 3],
        'classes_num': 10,
        'num_client': 100
    }
    dataset_and_model_har = {
        'dataset_name': 'HAR',
        'model_name': 'LR',
        'input_shape': [None, 561],
        'classes_num': 6,
        'target_label': 2,
        'num_client': 40
    }
    dataset_and_model_bcw = {
        'dataset_name': 'BCW',
        'model_name': 'LR',
        'input_shape': [None, 9],
        'classes_num': 2,
        'num_client': 20
    }
    parameter = dict()
    if dataset_name == 'MNIST':
        parameter['dataset_name'] = dataset_and_model_mnist['dataset_name']
        parameter['model_name'] = dataset_and_model_mnist['model_name']
        parameter['input_shape'] = dataset_and_model_mnist['input_shape']
        parameter['classes_num'] = dataset_and_model_mnist['classes_num']
        parameter['num_client'] = dataset_and_model_mnist['num_client']
    elif dataset_name == 'FMNIST':
        parameter['dataset_name'] = dataset_and_model_fmnist['dataset_name']
        parameter['model_name'] = dataset_and_model_fmnist['model_name']
        parameter['input_shape'] = dataset_and_model_fmnist['input_shape']
        parameter['classes_num'] = dataset_and_model_fmnist['classes_num']
        parameter['num_client'] = dataset_and_model_fmnist['num_client']
    elif dataset_name == 'CH-MNIST':
        parameter['dataset_name'] = dataset_and_model_chmnist['dataset_name']
        parameter['model_name'] = dataset_and_model_chmnist['model_name']
        parameter['input_shape'] = dataset_and_model_chmnist['input_shape']
        parameter['classes_num'] = dataset_and_model_chmnist['classes_num']
        parameter['num_client'] = dataset_and_model_chmnist['num_client']
    elif dataset_name == 'cifar-10':
        parameter['dataset_name'] = dataset_and_model_cifar10['dataset_name']
        parameter['model_name'] = dataset_and_model_cifar10['model_name']
        parameter['input_shape'] = dataset_and_model_cifar10['input_shape']
        parameter['classes_num'] = dataset_and_model_cifar10['classes_num']
        parameter['num_client'] = dataset_and_model_cifar10['num_client']
    elif dataset_name == 'HAR':
        parameter['dataset_name'] = dataset_and_model_har['dataset_name']
        parameter['model_name'] = dataset_and_model_har['model_name']
        parameter['input_shape'] = dataset_and_model_har['input_shape']
        parameter['classes_num'] = dataset_and_model_har['classes_num']
        parameter['num_client'] = dataset_and_model_har['num_client']
    elif dataset_name == 'BCW':
        parameter['dataset_name'] = dataset_and_model_bcw['dataset_name']
        parameter['model_name'] = dataset_and_model_bcw['model_name']
        parameter['input_shape'] = dataset_and_model_bcw['input_shape']
        parameter['classes_num'] = dataset_and_model_bcw['classes_num']
        parameter['num_client'] = dataset_and_model_bcw['num_client']
    else:
        assert 1, 'dataset name error'
    parameter['non_iid_p'] = non_iid_p
    parameter['malicious_clients_p'] = malicious_clients_p
    parameter['aggregation_method'] = aggregation_method
    parameter['defense_method'] = defense_method
    parameter['attack_mothod'] = attack_mothod
    return parameter


def train_discriminator_and_generator(really_list, discriminator, generator):
    discriminator_and_generator_train_datasets = tf.data.Dataset.from_tensor_slices(np.array(really_list)).shuffle(
        8).batch(256)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    for _ in tqdm(range(5000)):
        for data in zip(discriminator_and_generator_train_datasets):
            images = data[0]
            noises = tf.random.normal([len(images), 100])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noises, training=True)

                real_output = discriminator(images, training=True)
                fake_output = discriminator(generated_images,
                                            training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return discriminator, generator

def get_datas(data_name):
    if data_name == 'MNIST':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train / 255
        X_test = X_test / 255
        mean = 0.1307
        std = 0.3081
        X_train = ((X_train - mean) / std).reshape(-1, 28, 28, 1)
        X_test = ((X_test - mean) / std).reshape(-1, 28, 28, 1)
    elif data_name == 'FMNIST':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train / 255
        X_test = X_test / 255
        mean = 0.286
        std = 0.353
        X_train = ((X_train - mean) / std).reshape(-1, 28, 28, 1)
        X_test = ((X_test - mean) / std).reshape(-1, 28, 28, 1)
    elif data_name == 'CH-MNIST':
        features = np.load('datasets/CH-MNIST-features.npy', allow_pickle=True)
        labels = np.load('datasets/CH-MNIST-labels.npy', allow_pickle=True)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=66)
        X_train = X_train / 255
        X_test = X_test / 255
        mean = [0.6515, 0.475, 0.5866]
        std = [0.2538, 0.3266, 0.2667]
        for i in range(3):
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean[i]) / std[i]
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean[i]) / std[i]
    elif data_name == 'cifar-10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train = X_train / 255
        X_test = X_test / 255
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2070, 0.2435, 0.2616]
        y_train = y_train.reshape(-1, )
        for i in range(3):
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean[i]) / std[i]
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean[i]) / std[i]
    elif data_name == 'HAR':
        X_train = np.load('datasets/HAR-X-train.npy', allow_pickle=True)
        y_train = np.load('datasets/HAR-y-train.npy', allow_pickle=True).astype(np.int)
        X_test = np.load('datasets/HAR-X-test.npy', allow_pickle=True)
        y_test = np.load('datasets/HAR-y-test.npy', allow_pickle=True).astype(np.int)
    elif data_name == 'BCW':
        data = np.load('datasets/dataset-BCW.npy', allow_pickle=True)
        features = data[:, 0:-1]
        labels = data[:, -1].astype(np.int)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=66)
    else:
        return
    return X_train, X_test, y_train, y_test


def make_non_iid_datasets(X_train, y_train, p: float, clients_num, classes_num):
    index_by_group_id = [[] for _ in range(classes_num)]
    temp_pro = (1 - p) / (classes_num - 1)
    for i in range(len(X_train)):
        pro_list = []
        for j in range(classes_num):
            if j == y_train[i]:
                pro_list.append(p)
            else:
                pro_list.append(temp_pro)
        group_id = np.random.choice(classes_num, 1, p=pro_list)[0]
        index_by_group_id[group_id].append(i)

    nums_clients_by_every_groups = []
    basic_num = clients_num // classes_num
    remain_num = clients_num % classes_num
    for i in range(classes_num):
        if remain_num > 0:
            nums_clients_by_every_groups.append(basic_num + 1)
        else:
            nums_clients_by_every_groups.append(basic_num)
        remain_num -= 1

    client_datasets_X = []
    client_datasets_y = []
    for i in range(classes_num):  # i 表示第i组
        np.random.shuffle(index_by_group_id[i])
        client_data_basic_num = len(index_by_group_id[i]) // nums_clients_by_every_groups[i]
        remain_data_num = len(index_by_group_id[i]) % nums_clients_by_every_groups[i]
        index = 0
        for j in range(nums_clients_by_every_groups[i]):  # j表示第i组的第j个客户端
            if remain_data_num > 0:
                temp_client_data_len = client_data_basic_num + 1
            else:
                temp_client_data_len = client_data_basic_num
            remain_data_num -= 1

            temp_client_data_index = []
            for k in range(index, index + temp_client_data_len):  # k表示第i组第j个客户端的第k个实例
                temp_client_data_index.append(index_by_group_id[i][k])
            client_datasets_X.append(X_train[np.array(temp_client_data_index)])
            client_datasets_y.append(y_train[np.array(temp_client_data_index)])
            index += temp_client_data_len

    return client_datasets_X, client_datasets_y