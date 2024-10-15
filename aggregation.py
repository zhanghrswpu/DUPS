import numpy as np


def krum(clients_grads_list, malicious_num):
    client_num = len(clients_grads_list)
    non_malicious_count = client_num - malicious_num
    assert client_num > malicious_num

    dis_list = np.zeros((client_num, client_num), dtype=np.float)

    for i in range(client_num):
        for j in range(i):
            s = np.array(clients_grads_list[i], dtype=object) - np.array(clients_grads_list[j], dtype=object)
            ss = []
            for t in s:
                ss.append(t.reshape(-1, ))
            s = np.hstack(np.array(ss, dtype=object))
            dis_list[i][j] = dis_list[j][i] = np.linalg.norm(s) ** 2

    clients_grades = np.zeros((client_num), dtype=np.float)
    for i in range(client_num):
        temp_dis = np.sort(dis_list[i])
        clients_grades[i] = np.sum(temp_dis[:non_malicious_count])

    selected_clients_id = np.argmin(clients_grades)
    print('selected_clients_id:{}'.format(selected_clients_id))
    return selected_clients_id, clients_grads_list[selected_clients_id]
