import numpy as np
import matplotlib.pyplot as plt


def QKLMS_MSE(input_data, desired, order, delay, step_size, kernel_size, threshold, sqrt_power):
    N = 6
    element = input_data[0:order]
    center_list = np.zeros((1, order))
    center_list[0,:] = element

    alpha = np.zeros((delay,1))

    for k in range(delay):
        alpha[k] = np.array([step_size * desired[k]])

    length = input_data.shape[0] - order - delay
    e = np.zeros(length)
    y_pred = np.zeros(length)
    e[0] = desired[0]

    temp2 = np.zeros(delay)
    for step in range(1, length-delay, N):
        temp = np.zeros((delay, 1))

        if step >= (200 - order):
            input_data = np.hstack((input_data[:order + step],temp2[:N]))
        # if step >= (200 - order) and e_temp >= sqrt_power:
        #     print(step)
        #     break
        for k in range(delay):

            element = input_data[step:(step + order)]
            element = element.reshape(1, order)

            dist_list = ((center_list - element) ** 2).sum(1)

            Gs = np.exp(- kernel_size * dist_list)

            y_temp = np.dot(alpha[k,:], Gs)
            e_temp = desired[step+k] - y_temp

            dist_min = np.min(np.sqrt(dist_list))
            min_index = np.argmin(dist_list)

            e_temp = np.mean((e[step+k],e_temp))

            if dist_min <= threshold:
                a = step_size * e_temp
                alpha[k,min_index] = alpha[k,min_index] + a

            else:
                a = step_size * e_temp
                temp[k] = a
                if k == (delay -1):
                    center_list = np.vstack((center_list, element))
                    alpha = np.hstack((alpha, temp))
            temp2[k] = y_temp
            y_pred[step+k] = y_temp
            e[step+k] = e_temp


    return y_pred, e


def QKLMS_MCC(input_data, desired, order, delay, step_size, kernel_size, kernel_size2, threshold, sqrt_power):
    N = 6

    learning_rate = step_size * kernel_size2 * 2

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
    alpha = np.zeros((delay,1))
    for k in range(delay):
        G2_0 = np.exp(-0.5 * kernel_size2 * e[k] ** 2)
        alpha[k] = np.array([learning_rate * G2_0 * e[k]])

    temp2 = np.zeros(delay)

    for step in range(1, length-delay, N):
        temp = np.zeros((delay, 1))
        if step >= (200 - order):
            save_matrix[step] = np.hstack((save_matrix[step - 1], temp2[0]))[1:]
            # if e_temp >= sqrt_power:
            #     print(step)
            #     break
        for k in range(delay):
            element = save_matrix[step]

            dist_list = ((center_list - element) ** 2).sum(1)

            Gs = np.exp(- kernel_size * dist_list)
            y_temp = np.dot(alpha[k,:], Gs)
            e_temp = desired[step+k] - y_temp

            dist_min = np.min(np.sqrt(dist_list))
            min_index = np.argmin(dist_list)

            e_temp = np.mean((e[step+k], e_temp))

            if dist_min <= threshold:

                G2 = np.exp(-0.5 * kernel_size2 * e_temp ** 2)
                a = learning_rate * G2 * e_temp
                alpha[k,min_index] = alpha[k,min_index] + a

            else:
                G2 = np.exp(-0.5 * kernel_size2 * e_temp ** 2)
                a = learning_rate * G2 * e_temp
                temp[k] = a
                if k == (delay-1):
                    center_list = np.vstack((center_list, element))
                    alpha = np.hstack((alpha, temp))
            temp2[k] = y_temp
            y_pred[step+k] = y_temp
            e[step+k] = e_temp

    return y_pred, e


def QKRLS_MSE(input_data, desired, order, delay, lamda, forget_factor, kernel_size, threshold, sqrt_power):
    N = 1

    length = input_data.shape[0] - order - delay
    e = np.zeros(length)
    y_pred = np.zeros(length)

    save_matrix = np.zeros((length, order))
    for i in range(length):
        save_matrix[i] = input_data[i:i + order]

    center_list = np.zeros((1, order))
    center_list[0] = save_matrix[0]

    y_hat = np.zeros((delay,1))
    a_last = np.zeros((delay,1))

    p_last = 1 / ((np.array((lamda * forget_factor + 1)).reshape(1, 1)))
    for k in range(delay):
        y_hat[k] = desired[k]
        a_last[k] = p_last * desired[k]

    one = np.ones(1).reshape(1, 1)
    A = one

    temp5 = np.zeros(delay)
    for step in range(1, length-delay, N):
        temp4 = np.zeros((delay, 1))
        a_last_temp = a_last
        temp6 = np.zeros((delay,1))
        print(step)
        if step >= (200 - order):
            save_matrix[step] = np.hstack((save_matrix[step - 1], temp5[0]))[1:]
            # if e_temp >= sqrt_power:
            #     print(step)
            #     break
        for k in range(delay):
            element = save_matrix[step]
            dist_list = ((center_list - element) ** 2).sum(1)
            dist_min = np.min(dist_list)
            min_index = np.argmin(dist_list)

            if dist_min <= threshold:
                A_temp2 = A
                p_temp2 = p_last

                z = np.zeros((center_list.shape[0], center_list.shape[0]))
                z[min_index, min_index] = 1
                A = A + z
                K = np.exp(-kernel_size * ((center_list[:] - center_list[min_index]) ** 2).sum(1)).reshape(-1, 1)

                P = p_last[:, min_index].reshape(-1, 1)

                temp3 = P @ (K.T @ p_last) / (1 + K.T @ P)
                p_last = p_last - temp3
                y_hat[k,min_index] = y_hat[k,min_index] + desired[step+k]

                a_last[k] = (p_last @ y_hat[k,:]).reshape(-1, )
                y_temp = np.dot(K.T, a_last[k])
                e_temp = desired[step+k] - y_temp

                e_temp = np.mean((e[step+k],e_temp[0]))
                if k != (delay-1):
                    A = A_temp2
                    p_last = p_temp2
            else:

                p_temp = p_last
                A_temp2 = A

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

                y_temp = np.dot(G.T, a_last[k])
                e_temp = desired[step+k] - y_temp
                e_temp = np.mean((e[step + k], e_temp))

                temp2 = a_last[k].reshape(-1,1) - Z2 / r * e_temp

                a_last_temp[k] = temp2.reshape(-1,)
                temp6[k] = e_temp / r
                temp4[k] = desired[step+k]
                if k == (delay-1):
                    center_list = np.vstack((center_list, element))
                    y_hat = np.hstack((y_hat, temp4))
                    a_last = np.hstack((a_last_temp, temp6))


                if k != (delay-1):
                    p_last = p_temp

                    A = A_temp2
            temp5[k] = y_temp
            y_pred[step+k] = y_temp
            e[step+k] = e_temp
    return y_pred, e


def QKRLS_MCC(input_data, desired, order, delay, lamda, forget_factor, kernel_size, kernel_size2, threshold,
              sqrt_power):
    N = 1

    length = input_data.shape[0] - order - delay
    e = np.zeros(length)
    y_pred = np.zeros(length)

    save_matrix = np.zeros((length, order))
    for i in range(length):
        save_matrix[i] = input_data[i:i + order]

    center_list = np.zeros((1, order))
    center_list[0] = save_matrix[0]
    y_hat = np.zeros((delay,1))
    a_last = np.zeros((delay,1))
    p_last = 1 / ((np.array((lamda * forget_factor * kernel_size2 ** 2 + 1)).reshape(1, 1)))

    for k in range(delay):
        y_hat[k] = desired[k]
        a_last[k] = p_last * desired[k]

    one = np.ones(1).reshape(1, 1)
    A = one

    temp5 = np.zeros(delay)
    for step in range(1, 500, N):
        temp4 = np.zeros((delay,1))
        a_last_temp = a_last
        temp6 = np.zeros((delay,1))
        print(step)
        if step >= (200 - order):
            save_matrix[step] = np.hstack((save_matrix[step - 1], temp5[0]))[1:]
            # if e_temp >= sqrt_power:
            #     print(step)
            #     break
        for k in range(delay):
            element = save_matrix[step]
            dist_list = ((center_list - element) ** 2).sum(1)
            dist_min = np.min(dist_list)
            min_index = np.argmin(dist_list)

            if dist_min <= threshold:
                A_temp2 = A
                p_temp2 = p_last

                z = np.zeros((center_list.shape[0], center_list.shape[0]))
                z[min_index, min_index] = 1
                A = A + z
                K = np.exp(-kernel_size * ((center_list[:] - center_list[min_index]) ** 2).sum(1)).reshape(-1, 1)

                P = p_last[:, min_index].reshape(-1, 1)

                temp3 = P @ (K.T @ p_last) / (1 + K.T @ P)
                p_last = p_last - temp3
                y_hat[k,min_index] = y_hat[k,min_index] + desired[step+k]
                a_last[k] = (p_last @ y_hat[k,:]).reshape(-1,)
                y_temp = np.dot(K.T, a_last[k])
                e_temp = desired[step+k] - y_temp

                e_temp = np.mean((e[step+k],e_temp[0]))
                if k != (delay-1):
                    A = A_temp2
                    p_last = p_temp2

            else:
                p_temp = p_last
                a_temp = a_last[k]
                A_temp2 = A

                G = np.exp(-kernel_size * dist_list).reshape(-1, 1)
                y_temp = np.dot(G.T, a_last[k])
                e_temp = desired[step+k] - y_temp

                e_temp = np.mean((e[step+k],e_temp))

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

                temp2 = a_last[k].reshape(-1,1) - Z2 / r * e_temp
                a_last_temp[k] = temp2.reshape(-1,)
                temp6[k] = e_temp / r
                temp4[k] = desired[step+k]

                if k == (delay-1):
                    center_list = np.vstack((center_list, element))
                    y_hat = np.hstack((y_hat, temp4))
                    a_last = np.hstack((a_last_temp,temp6))
                if k != (delay-1):
                    p_last = p_temp
                    A = A_temp2
            temp5[k] = y_temp
            y_pred[step+k] = y_temp
            e[step+k] = e_temp

    return y_pred, e

if __name__ == '__main__':
    filtpath = "./sun_spot_input.asc"
    input_data = np.loadtxt(filtpath, skiprows=1)

    # power = (input_data ** 2).sum() / input_data.shape[0]
    # sqrt_power2 = np.sqrt(power)
    # input_data = input_data / sqrt_power2
    # sqrt_power = np.sqrt((input_data ** 2).sum() / input_data.shape[0])
    input_data = (input_data - min(input_data)) / (max(input_data) - min(input_data))

    sqrt_power = 0.3
    order = 6
    delay = 1
    desired = input_data[order + delay:]

    # 设置预测delay
    delay = 6

    # QKLMS_MSE
    step_size = 0.9
    kernel_size = 0.01
    threshold = 0

    # y_pred, e = QKLMS_MSE(input_data, desired, order, delay, step_size, kernel_size, threshold, sqrt_power)
    # plt.plot(y_pred, label="y_pred")
    # plt.plot(desired, label="desired")
    # plt.legend()
    # plt.show()

    # QKlms-mcc
    # step_size = 0.9
    # kernel_size2 = 0.1
    # kernel_size = 0.1
    # threshold = 0
    #
    # y_pred, e = QKLMS_MCC(input_data, desired, order, delay, step_size, kernel_size, kernel_size2, threshold,
    #                       sqrt_power)
    # plt.plot(y_pred, label="y_pred")
    # plt.plot(desired, label="desired")
    # plt.legend()
    # plt.show()

    # QKRLS-MSE
    lamda = 0.9
    forget_factor = 1
    kernel_size = 0.1
    threshold = 0.1

    # y_pred, e = QKRLS_MSE(input_data, desired, order, delay, lamda, forget_factor, kernel_size, threshold,sqrt_power)
    # plt.plot(y_pred[:500], label='y_pred')
    # plt.plot(desired[:500], label='desired')
    # plt.legend()
    # plt.show()

    # QKRLS-MCC
    lamda = 0.9
    forget_factor = 1
    kernel_size = 0.1
    kernel_size2 = 0.1
    threshold = 0.1

    y_pred, e = QKRLS_MCC(input_data, desired, order, delay, lamda, forget_factor, kernel_size, kernel_size2, threshold,sqrt_power)
    plt.plot(y_pred[:500], label='y_pred')
    plt.plot(desired[:500], label='desired')
    plt.legend()
    plt.show()