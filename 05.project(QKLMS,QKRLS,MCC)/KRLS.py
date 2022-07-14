import numpy as np
import matplotlib.pyplot as plt


def LMS_MSE(input_data, desired, order, delay, step_size):
    length = input_data.shape[0] - order - delay
    y_pred = np.zeros(length)
    e = np.zeros(length)
    w = np.zeros((length, order))

    for i in range(length):
        if i == 0:
            w_temp = np.random.normal(loc=0, scale=1, size=order)
        input_seg = input_data[i:i + order]
        input_seg = input_seg[::-1]
        y_temp = np.dot(w_temp, input_seg)
        e_temp = desired[i] - y_temp
        w_temp = w_temp + step_size * e_temp * input_seg
        y_pred[i] = y_temp
        e[i] = e_temp
        w[i, :] = w_temp.T
    return y_pred, e


def LMS_MCC(input_data, desired, order, delay, step_size, kernel_size2, L):
    learning_rate = step_size * kernel_size2 * 2

    length = input_data.shape[0] - order - delay
    y_pred = np.zeros(length)
    e = np.zeros(length)
    w = np.zeros((length, order))
    w_temp = np.random.normal(loc=0, scale=0, size=order)

    save_matrix = np.zeros((length, order))
    for i in range(length):
        save_matrix[i] = input_data[i:i + order]

    for i in range(L, length):
        input_seg = save_matrix[i]
        input_seg = input_seg[::-1]
        y_temp = np.dot(w_temp, input_seg)
        e_temp = desired[i] - y_temp

        G2 = np.exp(-kernel_size2 * e[i - L:i] ** 2)
        print(G2)
        temp = ((G2 * e[i - L:i]).reshape(-1, 1) * save_matrix[i - L:i][::-1]).sum(0)

        w_temp = w_temp + learning_rate * temp
        y_pred[i] = y_temp
        e[i] = e_temp
        w[i, :] = w_temp.T
    return y_pred, e


def QKLMS_MSE(input_data, desired, order, delay, step_size, kernel_size, threshold):
    count = order
    count_list = np.zeros([count])

    element = input_data[0:order]
    center_list = np.zeros((1, order))
    center_list[0] = element

    alpha = np.array([step_size * desired[order]])

    result = np.zeros(1)

    length = input_data.shape[0] - order - delay
    e = np.zeros(length)
    y_pred = np.zeros(length)
    e_temp = desired[order]
    e[0] = e_temp

    for step in range(1, length):

        element = input_data[step:(step + order)]
        element = element.reshape(1, order)
        dist_list = ((center_list - element) ** 2).sum(1)
        dist_min = np.min(np.sqrt(dist_list))
        min_index = np.argmin(dist_list)

        if dist_min <= threshold:
            a = step_size * e[step - 1]
            alpha[min_index] = alpha[min_index] + a
            Gs = np.exp(- kernel_size * ((center_list - center_list[min_index]) ** 2).sum(1))

            y_temp = np.dot(alpha, Gs)
            e_temp = desired[step] - y_temp
        else:

            count = count + 1
            Gs = np.exp(- kernel_size * dist_list)

            y_temp = np.dot(alpha, Gs)
            result = np.hstack((result, y_temp))
            e_temp = desired[step] - y_temp
            center_list = np.vstack((center_list, element))
            a = step_size * e_temp
            alpha = np.hstack((alpha, a))
        y_pred[step] = y_temp
        e[step] = e_temp
        count_list = np.hstack((count_list, count))

    print(count)
    return y_pred, e


def QKLMS_MCC(input_data, desired, order, delay, step_size, kernel_size, kernel_size2, threshold):
    learning_rate = step_size * kernel_size2 * 2
    count = order
    count_list = np.zeros([count])
    length = input_data.shape[0] - order - delay
    e = np.zeros(length)
    y_pred = np.zeros(length)
    e_temp = desired[0]
    e[0] = e_temp

    save_matrix = np.zeros((length, order))
    for i in range(length):
        save_matrix[i] = input_data[i:i + order]

    center_list = np.zeros((1, order))
    center_list[0, :] = save_matrix[0]
    G2_0 = np.exp(-0.5 * kernel_size2 * e[0] ** 2)
    alpha = np.array([learning_rate * G2_0 * e[0]])

    for step in range(1, length):

        element = input_data[step:(step + order)]
        element = element.reshape(1, order)
        dist_list = ((center_list - element) ** 2).sum(1)

        Gs = np.exp(- kernel_size * dist_list)
        y_temp = np.dot(alpha, Gs)
        e_temp = desired[step] - y_temp

        dist_min = np.min(np.sqrt(dist_list))
        min_index = np.argmin(dist_list)

        if dist_min <= threshold:

            G2 = np.exp(-0.5 * kernel_size2 * e_temp ** 2)
            a = learning_rate * G2 * e_temp
            alpha[min_index] = alpha[min_index] + a

        else:

            count = count + 1
            center_list = np.vstack((center_list, element))
            G2 = np.exp(-0.5 * kernel_size2 * e_temp ** 2)
            a = learning_rate * G2 * e_temp
            alpha = np.hstack((alpha, a))

        y_pred[step] = y_temp
        e[step] = e_temp
        count_list = np.hstack((count_list, count))

    print(count)
    return y_pred, e


def KRLS_MSE(input_data, desired, order, delay, lamda, forget_factor, kernel_size):
    length = input_data.shape[0] - order - delay
    e = np.zeros(length)
    y_pred = np.zeros(length)
    one = np.ones(1).reshape(1, 1)
    Q_last = 1 / ((np.array((lamda * forget_factor + 1)).reshape(1, 1)))
    a_last = Q_last * desired[0]

    save_matrix = np.zeros((length, order))
    for i in range(length):
        save_matrix[i] = input_data[i:i + order]

    for step in range(1, length):
        print(step)
        input_seg = save_matrix[step, :]
        G = np.exp(-kernel_size * ((save_matrix[:step] - input_seg) ** 2).sum(1)).reshape(-1, 1)

        Z = Q_last @ G

        r = (lamda * (forget_factor ** step) + 1 - Z.T @ G)
        Q_last = Q_last * r + Z * Z.T

        Q_last = np.hstack((Q_last, -Z))

        temp = np.hstack((-Z.T, one))

        Q_last = np.vstack((Q_last, temp))

        Q_last = Q_last / r
        y_pred[step] = np.dot(G.T, a_last)
        e[step] = desired[step] - y_pred[step]

        temp2 = a_last - Z / r * e[step]
        a_last = np.vstack((temp2, e[step] / r))

    return y_pred, e


def QKRLS_MSE(input_data, desired, order, delay, lamda, forget_factor, kernel_size, threshold):
    length = input_data.shape[0] - order - delay
    e = np.zeros(length)
    y_pred = np.zeros(length)

    save_matrix = np.zeros((length, order))
    for i in range(length):
        save_matrix[i] = input_data[i:i + order]

    count = order + delay
    count_list = np.zeros([count])

    center_list = np.zeros((1, order))
    center_list[0] = save_matrix[0]
    y_hat = np.zeros(1)
    y_hat[0] = desired[0]

    p_last = 1 / ((np.array((lamda * forget_factor + 1)).reshape(1, 1)))
    a_last = p_last * desired[0]

    one = np.ones(1).reshape(1, 1)
    A = one

    for step in range(1, length):
        print(step)
        element = save_matrix[step]
        dist_list = ((center_list - element) ** 2).sum(1)
        dist_min = np.min(dist_list)
        min_index = np.argmin(dist_list)

        if dist_min <= threshold:
            z = np.zeros((center_list.shape[0], center_list.shape[0]))
            z[min_index, min_index] = 1
            A = A + z
            K = np.exp(-kernel_size * ((center_list[:] - center_list[min_index]) ** 2).sum(1)).reshape(-1, 1)

            P = p_last[:, min_index].reshape(-1, 1)

            temp3 = P @ (K.T @ p_last) / (1 + K.T @ P)
            p_last = p_last - temp3
            y_hat[min_index] = y_hat[min_index] + desired[step]
            a_last = (p_last @ y_hat).reshape(-1, 1)
            y_temp = np.dot(K.T, a_last)
            e_temp = desired[step] - y_temp

        else:

            # p_temp = p_last
            # a_temp = a_last
            count = count + 1

            G = np.exp(-kernel_size * dist_list).reshape(-1, 1)

            Z1 = p_last.T @ G
            Z2 = p_last @ A @ G

            r = (lamda * (forget_factor ** step) + 1 - Z2.T @ G)

            p_last = p_last * r + Z2 * Z1.T

            p_last = np.hstack((p_last, -Z2))

            temp = np.hstack((-Z1.T, one))
            p_last = np.vstack((p_last, temp))

            p_last = p_last / r

            A_temp = np.zeros(p_last.shape)

            A_temp[:-1, :-1] = A
            A_temp[-1, -1] = 1
            A = A_temp

            y_temp = np.dot(G.T, a_last)
            e_temp = desired[step] - y_temp

            temp2 = a_last - Z2 / r * e_temp

            a_last = np.vstack((temp2, e_temp / r))
            center_list = np.vstack((center_list, element))
            y_hat = np.hstack((y_hat, desired[step]))

        y_pred[step] = y_temp
        e[step] = e_temp
    print(count)
    return y_pred, e


def QKRLS_MCC(input_data, desired, order, delay, lamda, forget_factor, kernel_size, kernel_size2, threshold):
    length = input_data.shape[0] - order - delay
    e = np.zeros(length)
    y_pred = np.zeros(length)

    save_matrix = np.zeros((length, order))
    for i in range(length):
        save_matrix[i] = input_data[i:i + order]

    count = order + delay
    count_list = np.zeros([count])

    center_list = np.zeros((1, order))
    center_list[0] = save_matrix[0]
    y_hat = np.zeros(1)
    y_hat[0] = desired[0]

    p_last = 1 / ((np.array((lamda * forget_factor * kernel_size2 ** 2 + 1)).reshape(1, 1)))

    a_last = p_last * desired[0]

    one = np.ones(1).reshape(1, 1)
    A = one

    for step in range(1, length):
        print(step)
        element = save_matrix[step]
        dist_list = ((center_list - element) ** 2).sum(1)
        dist_min = np.min(dist_list)
        min_index = np.argmin(dist_list)

        if dist_min <= threshold:
            z = np.zeros((center_list.shape[0], center_list.shape[0]))
            z[min_index, min_index] = 1
            A = A + z
            K = np.exp(-kernel_size * ((center_list[:] - center_list[min_index]) ** 2).sum(1)).reshape(-1, 1)

            P = p_last[:, min_index].reshape(-1, 1)

            temp3 = P @ (K.T @ p_last) / (1 + K.T @ P)
            p_last = p_last - temp3
            y_hat[min_index] = y_hat[min_index] + desired[step]
            a_last = (p_last @ y_hat).reshape(-1, 1)
            y_temp = np.dot(K.T, a_last)
            e_temp = desired[step] - y_temp

        else:

            count = count + 1

            G = np.exp(-kernel_size * dist_list).reshape(-1, 1)
            y_temp = np.dot(G.T, a_last)
            e_temp = desired[step] - y_temp
            G2 = np.exp(-0.5 * kernel_size2 * e_temp ** 2)

            Z1 = p_last.T @ G
            Z2 = p_last @ A @ G

            r = (lamda * (forget_factor ** step) * kernel_size2 * G2 + 1 - Z2.T @ G)

            p_last = p_last * r + Z2 * Z1.T

            p_last = np.hstack((p_last, -Z2))

            temp = np.hstack((-Z1.T, one))
            p_last = np.vstack((p_last, temp))

            p_last = p_last / r

            A_temp = np.zeros(p_last.shape)

            A_temp[:-1, :-1] = A
            A_temp[-1, -1] = 1
            A = A_temp

            temp2 = a_last - Z2 / r * e_temp

            a_last = np.vstack((temp2, e_temp / r))
            a = a_last
            center_list = np.vstack((center_list, element))
            y_hat = np.hstack((y_hat, desired[step]))

        y_pred[step] = y_temp
        e[step] = e_temp
    print(count)
    return y_pred, e


if __name__ == '__main__':
    filtpath = "./sun_spot_input.asc"
    input_data = np.loadtxt(filtpath, skiprows=1)
    print(input_data.shape)

    power = (input_data ** 2).sum() / input_data.shape[0]
    # plt.plot(input_data)
    # plt.show()

    order = 6
    delay = 1
    desired = input_data[order - delay:]
    # LMS-mse

    # print(1 / (order * power))
    step_size = 1 * 10 ** (-5)
    # print(step_size)

    # y_pred, e = LMS_MSE(input_data, desired, order, delay,step_size)
    # plt.plot(y_pred[0:200], label="y_pred")
    # plt.plot(desired[0:200], label="desired")
    # plt.legend()
    # plt.show()

    # square_e = e ** 2
    # temp = 0
    # mse_list = np.zeros(e.shape[0])
    # for i in range(y_pred.shape[0]):
    #     temp = (temp * i + square_e[i]) / (i + 1)
    #     mse_list[i] = temp
    # plt.plot(mse_list)
    # plt.ylim(0, 1000)
    # plt.show()

    # lms-mcc
    step_size = 0.9 * 10 ** (-3)
    # print(step_size)
    L = 10
    kernel_size2 = 0.001

    # y_pred, e = LMS_MCC(input_data, desired, order, delay, step_size,kernel_size2,L)
    # plt.plot(y_pred, label="y_pred")
    # plt.plot(desired, label="desired")
    # plt.legend()
    # plt.show()

    # square_e = e ** 2
    # temp = 0
    # mse_list = np.zeros(e.shape[0])
    # for i in range(y_pred.shape[0]):
    #     temp = (temp * i + square_e[i]) / (i + 1)
    #     mse_list[i] = temp
    # plt.plot(mse_list)
    # plt.ylim(0, 1000)
    # plt.show()

    # QKLMS-MSE
    step_size = 0.9
    kernel_size = 0.0001
    threshold = 0


    # y_pred, e = QKLMS_MSE(input_data, desired, order, delay, step_size, kernel_size, L)
    # plt.plot(y_pred[0:200], label="y_pred")
    # plt.plot(desired[0:200], label="desired")
    # plt.legend()
    # plt.show()



    # QKlms-mcc
    step_size = 4500
    kernel_size2 = 0.0001
    kernel_size = 0.0001
    threshold = 0

    # y_pred, e = QKLMS_MCC(input_data, desired, order, delay, step_size, kernel_size, kernel_size2, threshold)
    # plt.plot(y_pred[0:200], label="y_pred")
    # plt.plot(desired[0:200], label="desired")
    # plt.legend()
    # plt.show()

    # KRLS-MSE
    lamda = 0.1
    forget_factor = 1
    kernel_size = 0.0001
    kernel_size2 = 0.1
    threshold = 300

    # y_pred, e = KRLS_MSE(input_data, desired, order, delay, lamda, forget_factor, kernel_size)
    # y_pred, e = QKRLS_MSE(input_data, desired, order, delay, lamda, forget_factor, kernel_size, threshold)
    y_pred, e = QKRLS_MCC(input_data, desired, order, delay, lamda, forget_factor, kernel_size, kernel_size2, threshold)
    plt.plot(y_pred[0:200], label='y_pred')
    plt.plot(desired[0:200], label='desired')
    plt.legend()
    plt.show()

    # square_e = e ** 2
    # temp = 0
    # mse_list = np.zeros(e.shape[0])
    # for i in range(y_pred.shape[0]):
    #     temp = (temp * i + square_e[i]) / (i + 1)
    #     mse_list[i] = temp
    # plt.plot(mse_list)
    # plt.ylim(0, 1000)
    # plt.show()
