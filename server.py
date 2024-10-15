import numpy as np
from sklearn.model_selection import train_test_split
from model import *
from aggregation import *


class Server():
    def __init__(self, X_train, y_train, model_name, classes_num, malicious_clients_num, *input_shape):
        X_test, X_val, y_test, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=66)
        self.dataset = {
            'val_x': X_val,
            'val_y': y_val,
            'test_x': X_test,
            'test_y': y_test
        }
        self.malicious_clients_num = malicious_clients_num
        self.model = None
        if model_name == 'CNN':
            self.model = ConvolutionalNetwork()
        elif model_name == 'LR':
            self.model = LogisticRegression(classes_num)
        else:
            self.model = ResNet(classes_num)
        self.model.build(input_shape=input_shape)
        self.model.summary()

    def fedavg(self, clients_list):
        clients_model_grads_list = []
        for client in clients_list:
            clients_model_grads_list.append(client.grads)

        grads_list = []
        for grads in zip(*clients_model_grads_list):
            grads_list.append(np.mean(grads, axis=0))

        self.set_weights_by_grads(grads_list, self.model.get_weights())

    def krum(self, clients_list):
        clients_grads_list = []
        for client in clients_list:
            clients_grads_list.append(client.grads)
        _, grads = krum(clients_grads_list, self.malicious_clients_num)
        self.set_weights_by_grads(grads, self.model.get_weights())

    def trimmed_mean(self, clients_list):
        client_num = len(clients_list)
        assert self.malicious_clients_num < client_num / 2

        grads_list = []
        for i in range(len(clients_list[0].grads)):
            grads = np.zeros_like(clients_list[0].grads[i].flatten(), dtype=np.float)
            for j in range(len(grads)):
                temp_grads_list_index_i_layer = []
                for client in clients_list:
                    temp_client_grads_index_i_layer = client.grads[i]
                    temp_grads_list_index_i_layer.append(temp_client_grads_index_i_layer.flatten()[j])
                temp_grads_list_index_i_layer = np.sort(temp_grads_list_index_i_layer)[
                                                self.malicious_clients_num:client_num - self.malicious_clients_num]
                grads[j] = np.mean(temp_grads_list_index_i_layer)
            grads = grads.reshape(clients_list[0].grads[i].shape)
            grads_list.append(grads)

        self.set_weights_by_grads(grads_list, self.model.get_weights())

    def trimmed_median(self, clients_list):
        client_num = len(clients_list)
        assert self.malicious_clients_num < client_num / 2

        grads_list = []
        for i in range(len(clients_list[0].grads)):
            grads = np.zeros_like(clients_list[0].grads[i].flatten(), dtype=np.float)
            for j in range(len(grads)):
                temp_grads_list_index_i_layer = []
                for client in clients_list:
                    temp_client_grads_index_i_layer = client.grads[i]
                    temp_grads_list_index_i_layer.append(temp_client_grads_index_i_layer.flatten()[j])
                temp_grads_list_index_i_layer = np.sort(temp_grads_list_index_i_layer)[
                                                self.malicious_clients_num:client_num - self.malicious_clients_num]
                grads[j] = np.median(temp_grads_list_index_i_layer)
            grads = grads.reshape(clients_list[0].grads[i].shape)
            grads_list.append(grads)

        self.set_weights_by_grads(grads_list, self.model.get_weights())

    def err(self, clients_list, aggregation_method):
        X_val = self.dataset['val_x']
        y_val = self.dataset['val_y']
        data_num = len(X_val)
        client_num = len(clients_list)
        non_malicious_count = client_num - self.malicious_clients_num
        server_model_weights = self.model.get_weights()

        clients_grads_acc_list = np.zeros((client_num), dtype=np.float)
        for i in range(client_num):
            temp_clients_list = []
            for j in range(client_num):
                if i == j:
                    continue
                temp_clients_list.append(clients_list[j])

            self.model.set_weights(server_model_weights)
            self.aggregation(aggregation_method, temp_clients_list)

            temp_pre = tf.nn.softmax(self.model(X_val), -1)
            ac_count = 0
            for j in range(data_num):
                temp_pre_label = tf.argmax(temp_pre[j], -1)
                if temp_pre_label == y_val[j]:
                    ac_count += 1
            accuracy = ac_count / data_num
            clients_grads_acc_list[i] = accuracy

        client_index = np.argsort(clients_grads_acc_list)[::-1]
        non_malicious_client_index = client_index[:non_malicious_count]
        non_malicious_clients_list = []
        for i in non_malicious_client_index:
            non_malicious_clients_list.append(clients_list[i])

        self.model.set_weights(server_model_weights)
        self.aggregation(aggregation_method, non_malicious_clients_list)

    def lfr(self, clients_list, aggregation_method):
        X_val = self.dataset['val_x']
        y_val = self.dataset['val_y']
        client_num = len(clients_list)
        non_malicious_count = client_num - self.malicious_clients_num
        server_model_weights = self.model.get_weights()
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        clients_grads_loss_list = np.zeros((client_num), dtype=np.float)
        for i in range(client_num):
            temp_clients_list = []
            for j in range(client_num):
                if i == j:
                    continue
                temp_clients_list.append(clients_list[j])

            self.model.set_weights(server_model_weights)
            self.aggregation(aggregation_method, temp_clients_list)

            temp_pre = self.model(X_val)
            loss = scce(y_val, temp_pre)
            clients_grads_loss_list[i] = loss.numpy()

        client_index = np.argsort(clients_grads_loss_list)
        non_malicious_client_index = client_index[:non_malicious_count]
        non_malicious_clients_list = []
        for i in non_malicious_client_index:
            non_malicious_clients_list.append(clients_list[i])

        self.model.set_weights(server_model_weights)
        self.aggregation(aggregation_method, non_malicious_clients_list)

    def union(self, clients_list, aggregation_method):
        X_val = self.dataset['val_x']
        y_val = self.dataset['val_y']
        client_num = len(clients_list)
        non_malicious_count = client_num - self.malicious_clients_num
        data_num = len(X_val)
        server_model_weights = self.model.get_weights()
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        clients_grads_loss_list = np.zeros((client_num), dtype=np.float)
        clients_grads_acc_list = np.zeros((client_num), dtype=np.float)
        for i in range(client_num):
            temp_clients_list = []
            for j in range(client_num):
                if i == j:
                    continue
                temp_clients_list.append(clients_list[j])

            self.model.set_weights(server_model_weights)
            self.aggregation(aggregation_method, temp_clients_list)

            temp_pre = self.model(X_val)
            loss = scce(y_val, temp_pre)
            clients_grads_loss_list[i] = loss.numpy()

            temp_pre = tf.nn.softmax(temp_pre, -1)
            ac_count = 0
            for j in range(data_num):
                temp_pre_label = tf.argmax(temp_pre[j], -1)
                if temp_pre_label == y_val[j]:
                    ac_count += 1
            accuracy = ac_count / data_num
            clients_grads_acc_list[i] = accuracy

        client_index_by_loss = np.argsort(clients_grads_loss_list)
        non_malicious_client_index_by_loss = client_index_by_loss[:non_malicious_count]
        client_index_by_acc = np.argsort(clients_grads_acc_list)[::-1]
        non_malicious_client_index_by_acc = client_index_by_acc[:non_malicious_count]

        client_index = np.intersect1d(non_malicious_client_index_by_acc, non_malicious_client_index_by_loss)
        non_malicious_client_index = client_index[:non_malicious_count]
        non_malicious_clients_list = []
        for i in non_malicious_client_index:
            non_malicious_clients_list.append(clients_list[i])

        self.model.set_weights(server_model_weights)
        self.aggregation(aggregation_method, non_malicious_clients_list)

    def server_model_test(self, classes_num, target_label):
        X_test = self.dataset['test_x']
        y_test = self.dataset['test_y']
        data_num = len(X_test)

        temp_pre = tf.nn.softmax(self.model(X_test), -1)
        confusion_matrix = np.zeros((classes_num, classes_num), dtype=np.int)
        for i in range(data_num):
            temp_pre_label = tf.argmax(temp_pre[i], -1)
            confusion_matrix[int(y_test[i])][int(temp_pre_label)] += 1

        error_num = np.sum(confusion_matrix) - np.sum(np.array([confusion_matrix[i, i] for i in range(classes_num)]))
        misclassified_as_target_label_num = np.sum(confusion_matrix[:, target_label]) - confusion_matrix[
            target_label, target_label]
        accuracy = (data_num - error_num) / data_num
        posison_accuracy = misclassified_as_target_label_num / (data_num - np.sum(confusion_matrix[target_label, :]))
        return accuracy, posison_accuracy

    def set_weights_by_grads(self, grads_list, server_model_weights):
        new_server_weights = []
        for j in range(len(grads_list)):
            new_server_weights.append(server_model_weights[j] + 1 * grads_list[j])
        self.model.set_weights(new_server_weights)

    def aggregation(self, aggregation_method, clients_list):
        if aggregation_method == 'fedavg':
            self.fedavg(clients_list)
        elif aggregation_method == 'krum':
            self.krum(clients_list)
        elif aggregation_method == 'mean':
            self.trimmed_mean(clients_list)
        elif aggregation_method == 'median':
            self.trimmed_median(clients_list)
        else:
            assert 1 == 0, 'aggregation method error'