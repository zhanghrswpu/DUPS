import copy

import numpy as np

from model import *
from aggregation import *
from utils import *


def attack_by_scaling(clients_list, classes_num, model_weights, model_name, target_label, true_label, *input_shape):
    for client in clients_list:
        X_train = client.datasets['x']
        y_train = client.datasets['y']

        # Make sample backdoor
        malicious_label_list = []
        for i in range(len(y_train)):
            if y_train[i] == true_label:
                malicious_label = target_label
            else:
                malicious_label = y_train[i]
            malicious_label_list.append(malicious_label)
        malicious_label_list = np.array(malicious_label_list)

        # Initialize the client model
        client.model_init(model_name, classes_num, model_weights, *input_shape)

        client.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        client.model.fit(X_train, malicious_label_list, epochs=1, verbose=0, batch_size=8)

        new_model_weights = client.model.get_weights()
        temp_grads = []
        for i in range(len(new_model_weights)):
            temp_grads.append(new_model_weights[i] - model_weights[i])
        client.grads = temp_grads
        client.model = None
    return clients_list


def attack_by_gaussian_noise(clients_list, server_model_weights, model_name, classes_num, *input_shape):
    clients_num = len(clients_list)
    clients_model_grads_list = []
    for client in clients_list:
        client.client_train(server_model_weights, model_name, classes_num, *input_shape)
        clients_model_grads_list.append(client.grads)

    grads_mean_list = []
    grads_std_list = []
    for grads in zip(*clients_model_grads_list):
        grads_mean_list.append(np.mean(grads, axis=0))
        grads_std_list.append(np.std(grads, axis=0))

    clients_malicious_grads_list = [[] for _ in range(clients_num)]
    for i in range(len(grads_mean_list)):  # i 表示第i层梯度
        temp_mean = grads_mean_list[i].flatten()
        temp_std = grads_std_list[i].flatten()
        for j in range(clients_num):  # j表示第j个恶意客户端
            temp_grads = np.zeros((len(temp_mean)), dtype=np.float)
            for k in range(len(temp_mean)):  # k表示第i层梯度的第k个梯度
                temp_grads[k] = np.random.normal(temp_mean[k], temp_std[k], 1)[0]
            temp_grads = temp_grads.reshape(grads_mean_list[i].shape)
            clients_malicious_grads_list[j].append(temp_grads)

    for i in range(clients_num):
        clients_list[i].grads = clients_malicious_grads_list[i]
    return clients_list


def attack_by_label_flipping(clients_list, classes_num, server_model_weights, model_name, *input_shape):
    for client in clients_list:
        X_train = client.datasets['x']
        y_train = client.datasets['y']

        # label_flipping
        malicious_label_list = []
        for i in range(len(y_train)):
            malicious_label = classes_num - int(y_train[i]) - 1
            malicious_label_list.append(malicious_label)
        malicious_label_list = np.array(malicious_label_list)

        # Initialize the client model
        client.model_init(model_name, classes_num, server_model_weights, *input_shape)

        client.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        client.model.fit(X_train, malicious_label_list, epochs=1, verbose=0, batch_size=8)

        new_model_weights = client.model.get_weights()
        temp_grads = []
        for i in range(len(new_model_weights)):
            temp_grads.append(new_model_weights[i] - server_model_weights[i])
        client.grads = temp_grads
        client.model = None
    return clients_list


def attack_by_krum(clients_list, server_model_weights, model_name, classes_num, *input_shape):
    bound = 1e-5
    upper = 1
    client_num = len(clients_list)

    malicious_client_grads_list = []
    for client in clients_list:
        client.client_train(server_model_weights, model_name, classes_num, *input_shape)
        malicious_client_grads_list.append(client.grads)

    dis_list_client_and_client = np.zeros((client_num, client_num), dtype=np.float)
    dis_list_client_and_server = np.zeros((client_num), dtype=np.float)
    weights_num = 0
    for i in range(client_num):
        for j in range(i):
            s = np.array(malicious_client_grads_list[i], dtype=object) - np.array(malicious_client_grads_list[j],
                                                                                  dtype=object)
            ss = []
            for t in s:
                ss.append(t.reshape(-1, ))
            s = np.hstack(np.array(ss, dtype=object))
            weights_num = len(s)
            dis_list_client_and_client[i][j] = dis_list_client_and_client[j][i] = np.linalg.norm(s) ** 2
        s = np.array(malicious_client_grads_list[i], dtype=object) - np.array(server_model_weights,
                                                                              dtype=object)
        ss = []
        for t in s:
            ss.append(t.reshape(-1, ))
        s = np.hstack(np.array(ss, dtype=object))
        dis_list_client_and_server[i] = np.linalg.norm(s)

    dis_sum_list_client = np.sum(np.sqrt(dis_list_client_and_client), -1)
    upper = np.min(dis_sum_list_client) / (len(clients_list) - 2 * client_num - 1) / np.sqrt(weights_num) + np.max(
        dis_list_client_and_server) / np.sqrt(weights_num)

    direction_list = []
    mean_grads_list = []
    for grads in zip(*malicious_client_grads_list):
        temp_grads = np.mean(grads, axis=0)
        temp_s = copy.deepcopy(temp_grads)
        temp_s[temp_s < 0] = -1
        temp_s[temp_s > 0] = 1
        direction_list.append(temp_s)
        mean_grads_list.append(temp_grads)

    count = 1
    while True:
        a = upper
        while a > bound:
            temp_grads = np.array(mean_grads_list, dtype=object) - a * np.array(direction_list, dtype=object)

            grads_list = []
            for i in range(count):
                grads_list.append(temp_grads)
            for malicious_client_grads in malicious_client_grads_list:
                grads_list.append(malicious_client_grads)

            selected_clients_id, _ = krum(grads_list, 0)
            if selected_clients_id < count:
                res = []
                for i in range(len(temp_grads)):
                    res.append(temp_grads[i])

                for client in clients_list:
                    client.grads = res
                return clients_list
            a *= 0.5
        count += 1


def attack_by_trimmed_mean(clients_list, server_model_weights, model_name, classes_num, *input_shape):
    clients_num = len(clients_list)
    malicious_client_grads_list = []
    for client in clients_list:
        client.client_train(server_model_weights, model_name, classes_num, *input_shape)
        malicious_client_grads_list.append(client.grads)

    grads_mean_list = []
    grads_std_list = []
    direction_list = []
    for grads in zip(*malicious_client_grads_list):
        temp_direction = np.mean(grads, axis=0)
        temp_direction[temp_direction < 0] = -1
        temp_direction[temp_direction > 0] = 1
        direction_list.append(temp_direction)

        grads_mean_list.append(np.mean(grads, axis=0))
        grads_std_list.append(np.std(grads, axis=0))

    clients_malicious_grads_list = [[] for _ in range(clients_num)]
    for i in range(len(grads_mean_list)):  # i 表示第i层梯度
        temp_mean = grads_mean_list[i].flatten()
        temp_std = grads_std_list[i].flatten()
        temp_direction = direction_list[i].flatten()
        for j in range(clients_num):  # j表示第j个恶意客户端
            temp_grads = np.zeros((len(temp_mean)), dtype=np.float)
            for k in range(len(temp_mean)):  # k表示第i层梯度的第k个梯度
                if temp_direction[k] == -1:
                    temp_grads[k] = np.random.uniform(temp_mean[k] + 3 * temp_std[k], temp_mean[k] + 4 * temp_std[k])
                elif temp_direction[k] == 1:
                    temp_grads[k] = np.random.uniform(temp_mean[k] - 4 * temp_std[k], temp_mean[k] - 3 * temp_std[k])
                else:
                    temp_grads[k] = 0
            temp_grads = temp_grads.reshape(grads_mean_list[i].shape)
            clients_malicious_grads_list[j].append(temp_grads)

    for i in range(clients_num):
        clients_list[i].grads = clients_malicious_grads_list[i]

    return clients_list


def attack_by_trimmed_median(clients_list, server_model_weights, model_name, classes_num, *input_shape):
    clients_num = len(clients_list)
    malicious_client_grads_list = []
    for client in clients_list:
        client.client_train(server_model_weights, model_name, classes_num, *input_shape)
        malicious_client_grads_list.append(client.grads)

    grads_max_list = []
    grads_min_list = []
    direction_list = []
    for grads in zip(*malicious_client_grads_list):
        temp_direction = np.mean(grads, axis=0)
        temp_direction[temp_direction < 0] = -1
        temp_direction[temp_direction > 0] = 1
        direction_list.append(temp_direction)

        grads_max_list.append(np.max(grads, axis=0))
        grads_min_list.append(np.min(grads, axis=0))

    clients_malicious_grads_list = [[] for _ in range(clients_num)]
    for i in range(len(grads_max_list)):  # i 表示第i层梯度
        temp_max = grads_max_list[i].flatten()
        temp_min = grads_min_list[i].flatten()
        temp_direction = direction_list[i].flatten()
        for j in range(clients_num):  # j表示第j个恶意客户端
            temp_grads = np.zeros((len(temp_max)), dtype=np.float)
            for k in range(len(temp_max)):  # k表示第i层梯度的第k个梯度
                if temp_direction[k] == -1:
                    temp_grads[k] = np.random.uniform(temp_max[k], 2 * temp_max[k])
                elif temp_direction[k] == 1:
                    temp_grads[k] = np.random.uniform(temp_min[k] / 2, temp_min[k])
                else:
                    temp_grads[k] = 0
            temp_grads = temp_grads.reshape(grads_max_list[i].shape)
            clients_malicious_grads_list[j].append(temp_grads)

    for i in range(clients_num):
        clients_list[i].grads = clients_malicious_grads_list[i]
    return clients_list


def attack_by_uncertain_samples(clients_list, server_model_weights, model_name, classes_num, true_label,
                                *input_shape):
    X_train = []
    y_train = []
    for client in clients_list:
        temp_X_train = client.datasets['x']
        temp_y_train = client.datasets['y']
        if len(X_train) == 0:
            X_train = temp_X_train
            y_train = temp_y_train
        else:
            X_train = np.append(X_train, temp_X_train, 0)
            y_train = np.append(y_train, temp_y_train, 0)

    # classifier initialization
    if model_name == 'CNN':
        classifier = ConvolutionalNetwork()
        classifier.build(input_shape=input_shape)
    elif model_name == 'LR':
        classifier = LogisticRegression(classes_num)
        classifier.build(input_shape=input_shape)
    else:
        classifier = ResNet(classes_num)
        classifier.build(input_shape=input_shape)
    classifier.set_weights(server_model_weights)

    # find the samples used to train the discriminator
    really_pre = tf.nn.softmax(classifier(X_train), -1)
    really_dataset_list = [[] for _ in range(classes_num)]
    really_len_list = np.zeros(classes_num, dtype=np.int)
    for i in range(len(really_pre)):
        if y_train[i] == true_label and tf.math.argmax(really_pre[i], -1) != true_label:
            really_dataset_list[tf.math.argmax(really_pre[i], -1)].append(X_train[i])
            really_len_list[tf.math.argmax(really_pre[i], -1)] += 1

    if np.max(really_len_list) > 0:
        target_label = np.argmax(really_len_list)
        X_train_really = really_dataset_list[target_label]

    else:
        print('If the sample is not selected, the original client gradient will not be changed')
        return clients_list

    fake_X_train = np.array(X_train_really)
    fake_y_train = np.zeros((len(fake_X_train)), dtype=np.int)
    for i in range(len(fake_X_train)):
        fake_y_train[i] = true_label
    classifier.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    classifier.fit(fake_X_train, fake_y_train, epochs=1, verbose=0, batch_size=1)
    new_model_weights = classifier.get_weights()
    temp_grads = []
    for i in range(len(new_model_weights)):
        temp_grads.append(1 * (new_model_weights[i] - server_model_weights[i]))
    for client in clients_list:
        client.grads = temp_grads

    return clients_list
