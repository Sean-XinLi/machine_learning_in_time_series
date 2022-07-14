import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, signal


def wiener_solution(input_data, desired, window_size, order):
    P_list = []
    R_list = []
    w_list = np.zeros(((input_data.shape[0] - window_size), order))
    for step in range(input_data.shape[0] - window_size):
        R = np.zeros(order)
        P = np.zeros(order)
        s1 = input_data[step:step + window_size]
        for tao in range(order):
            s2 = input_data[(step + tao):(step + tao + window_size)]
            s3 = desired[(step + tao):(step + tao + window_size)]
            R[tao] = (window_size - tao) / window_size * \
                     np.dot(s1[:(window_size - tao)], s2[:(window_size - tao)])
            P[tao] = (window_size - tao) / window_size * \
                     np.dot(s1[:(window_size - tao)], s3[:(window_size - tao)])
        R_inverse = np.linalg.inv(linalg.toeplitz(R))
        w = np.dot(R_inverse, P)
        P_list.append(P)
        R_list.append(R)
        w_list[step, :] = w
    sample_input = np.zeros((order, (input_data.shape[0] - order)))
    for step in range(input_data.shape[0] - order):
        temp = input_data[step:(step + order)]
        sample_input[:, step] = temp[::-1]
    y_pred = np.sum(w_list @ sample_input, axis=1)
    e = desired[order:order + y_pred.shape[0]] - y_pred
    return w_list, y_pred, e, P_list, R_list


def LMS(input_data, desired, step_size, order):
    length = input_data.shape[0] - order
    y_pred = np.zeros(length)
    e = np.zeros(length)
    w = np.zeros((length, order))

    for i in range(length):
        if i == 0:
            w_temp = np.random.normal(loc=0, scale=1, size=order)
        input_seg = input_data[i:i + order]
        input_seg = input_seg[::-1]
        y_temp = np.dot(w_temp, input_seg)
        e_temp = desired[i + order] - y_temp
        w_temp = w_temp + step_size * e_temp * input_seg

        y_pred[i] = y_temp
        e[i] = e_temp
        w[i, :] = w_temp.T
    return w, y_pred, e


def KLMS(input_data, desired, step_size, order, kernel_size):
    length = input_data.shape[0] - order
    e = np.zeros(length + 1)
    y_pred = np.zeros(length)
    e[0] = desired[0]

    save_matrix = np.zeros((length, order))
    for i in range(length):
        save_matrix[i] = input_data[i:i + order]
    for step in range(1, length):

        input_seg = input_data[step:(step + order)]
        input_seg = input_seg.reshape(1, order)
        y_temp = 0
        G = np.exp(-kernel_size * ((save_matrix[:step] - input_seg) ** 2).sum(1))
        y_temp = (step_size * e[:step] * G).sum()

        e_temp = desired[step] - y_temp
        e[step] = e_temp
        y_pred[step] = y_temp

    return y_pred, e, save_matrix


def QLMS(input_data, desired, step_size, order, kernel_size, threshold):
    count = order
    count_list = np.zeros([count])

    element = input_data[0:order]
    center_list = np.zeros((1, order))
    center_list[0] = element
    alpha = np.array([step_size * desired[order]])
    result = np.zeros(1)

    length = input_data.shape[0] - order
    e = np.zeros(length)
    y_pred = np.zeros(length)
    e_temp = desired[order]
    e[0] = e_temp

    for step in range(1, length):
        element = input_data[step:(step + order)]
        element = element.reshape(1, order)
        dist_list = ((center_list - element) ** 2).sum(1)
        dist_min = np.min(dist_list * 10 ** 29)
        min_index = np.argmin(dist_list)

        if dist_min <= threshold:
            a = step_size * e[step - 1]
            alpha[min_index] = alpha[min_index] + a
            Gs = np.exp(- kernel_size * dist_list)
            y_temp = np.dot(alpha, Gs)
            e_temp = desired[step + order] - y_temp
        else:
            count = count + 1
            Gs = np.exp(- kernel_size * dist_list)

            y_temp = np.dot(alpha, Gs)
            result = np.hstack((result, y_temp))
            e_temp = desired[step + order] - y_temp
            center_list = np.vstack((center_list, element))
            a = step_size * e_temp
            alpha = np.hstack((alpha, a))
        y_pred[step] = y_temp
        e[step] = e_temp
        count_list = np.hstack((count_list, count))

    print(count)
    return y_pred, e


def KLMS_MEE(input_data, desired, step_size, order, kernel_size, kernel_size2, o):
    length = input_data.shape[0] - order
    e = np.zeros(length)
    y_pred = np.zeros(length)
    e[0] = desired[0 + order]

    save_matrix = np.zeros((length, order))
    for i in range(length):
        save_matrix[i] = input_data[i:i + order]

    for step in range(1, 1000):
        print(step)
        e_ij = e[:step]
        e_ji = e_ij.reshape(-1, 1)
        z = e_ij - e_ji
        G2 = np.exp(-kernel_size2 * z ** 2)

        input_seg = input_data[step:(step + order)]
        input_seg = input_seg.reshape(1, order)

        G = np.exp(-kernel_size * ((save_matrix[:step] - input_seg) ** 2).sum(1))
        G_t = G.reshape(-1, 1)
        x = G - G_t
        a = (G2 * z * x).sum(0).sum()

        y_temp = step_size * 2 / step ** 2 * (G2 * z * x).sum(0).sum() + desired[step + order - o:step + order].mean()
        e[step] = desired[step + order] - y_temp
        y_pred[step] = y_temp

    return y_pred, e


def KLMS_MEE2(input_data, desired, step_size, order, kernel_size, kernel_size2, length_term):
    # initialize
    learning_rate = step_size / (window_size / kernel_size2 ** 2)
    # print(learning_rate)

    # start from 100th sample
    length = input_data.shape[0] - window_size - order
    y_pred = np.zeros(length)

    e_sum = np.zeros(input_data.shape[0] - order)

    save_matrix = np.zeros((length + window_size, order))
    for i in range(length):
        save_matrix[i] = input_data[i:i + order]
    e_mj = np.zeros(window_size)

    for step in range(window_size, length_term):
        print(step)

        x_i = save_matrix[step, :]
        term = np.zeros(step - window_size)

        for m in range(window_size, step):
            e_m = e_sum[m]

            x_m = save_matrix[m + 1, :]

            e_mj = e_sum[m - window_size:m]

            x_mj = save_matrix[m - window_size + 1:m + 1, :]

            z = e_m - e_mj

            G2 = np.exp(-kernel_size2 * z ** 2)

            G1_1 = np.exp(-kernel_size * ((x_i - x_mj) ** 2).sum(1))
            G1_2 = np.exp(-kernel_size * ((x_i - x_m) ** 2).sum())

            G1 = G1_1 - G1_2

            term[m - window_size] = (G2 * z * G1).sum()

        y_pred[step] = learning_rate * term.sum()

        e_sum[step] = -(desired[step + order] - y_pred[step])
    return y_pred, e_sum, save_matrix


# def KLMS_MME3

def prediction(input_data, save_matrix, e, step_size, order, kernel_size):
    length = input_data.shape[0] - order
    y_pred = np.zeros(length)
    for step in range(1, length):
        input_seg = input_data[step:(step + order)]
        input_seg = input_seg.reshape(1, order)
        y_temp = 0
        G = np.exp(-kernel_size * ((save_matrix[:step] - input_seg) ** 2).sum(1))
        y_temp = (step_size * e[:step] * G).sum()
        y_pred[step] = y_temp
    return y_pred

def prediction2(input_data, save_matrix, e_sum, step_size, order, kernel_size, kernel_size2, length_term):
    # initialize
    learning_rate = step_size / (window_size / kernel_size2 ** 2)
    # print(learning_rate)

    # start from 100th sample
    length = input_data.shape[0] - window_size - order
    y_pred = np.zeros(length)

    e_mj = np.zeros(window_size)

    for step in range(window_size, length_term):
        print(step)

        x_i = save_matrix[step, :]
        term = np.zeros(step - window_size)

        for m in range(window_size, step):
            e_m = e_sum[m]

            x_m = save_matrix[m + 1, :]

            e_mj = e_sum[m - window_size:m]

            x_mj = save_matrix[m - window_size + 1:m + 1, :]

            z = e_m - e_mj

            G2 = np.exp(-kernel_size2 * z ** 2)

            G1_1 = np.exp(-kernel_size * ((x_i - x_mj) ** 2).sum(1))
            G1_2 = np.exp(-kernel_size * ((x_i - x_m) ** 2).sum())

            G1 = G1_1 - G1_2

            term[m - window_size] = (G2 * z * G1).sum()

        y_pred[step] = learning_rate * term.sum()
    return y_pred


if __name__ == '__main__':
    # generate data
    # input freq
    freq1 = 1000
    # sample freq
    freq2 = 10000
    # desired freq
    freq3 = 2000
    T = 2
    t = np.arange(0, T, 1 / freq2)

    input_data = np.sin(2 * np.pi * freq1 * t)
    desired = np.sin(2 * np.pi * freq3 * t)

    # fig = plt.figure(figsize=(20, 8), dpi=100)
    # plt.subplot(2, 1, 1)
    # plt.specgram(input_data, Fs=freq2)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.subplot(2, 1, 2)
    # plt.tight_layout()
    # plt.specgram(desired, Fs=freq2)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.tight_layout()
    # plt.show()

    # plt.plot(t, input_data, label="input")
    # plt.plot(t, desired, label="desired")
    # plt.xlim(0, 0.01)
    # plt.legend(loc="upper right")
    # plt.title("input data and desired data in [0,0.01]s")
    #
    # plt.show()

    order = 10
    # wiener solution
    window_size = 100

    # w1, y1, e1, P, R = wiener_solution(input_data, desired, window_size, order)
    # plt.plot(y1, label='y_pred')
    # plt.plot(desired, label='desired')
    # plt.legend(loc="upper right")
    # plt.xlim(0, 200)
    # plt.title("performance of prediction signal in wiener solution")
    #
    # plt.show()

    # fig = plt.figure(figsize=(20, 8), dpi=100)
    # plt.subplot(2, 1, 1)
    # plt.specgram(y1, Fs=freq2)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.subplot(2, 1, 2)
    # plt.tight_layout()
    # plt.specgram(desired, Fs=freq2)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.tight_layout()
    # plt.show()

    # square_e = e1 ** 2
    # temp = 0
    # mse_list = np.zeros(y1.shape[0])
    # for i in range(y1.shape[0]):
    #     temp = (temp * i + square_e[i]) / (i + 1)
    #     mse_list[i] = temp
    # plt.plot(mse_list)
    # plt.title("performance of MSE in wiener solution")
    # plt.show()

    # for i in range(order):
    #     plt.plot(w1[:, i])
    # plt.xlim(0, 50)
    # plt.title("performance of parameter w in wiener solution in 50 samples")
    # plt.savefig("1.png")
    # plt.show()

    # LMS
    step_size = 0.18

    # w2, y2, e2 = LMS(input_data, desired, step_size, order)
    # plt.plot(y2, label='y_pred')
    # plt.plot(desired, label='desired')
    # plt.legend(loc="upper right")
    # plt.xlim(0, 200)
    # plt.title("performance of prediction signal in LMS")
    #
    # plt.show()

    # square_e = e2 ** 2
    # temp = 0
    # mse_list = np.zeros(y2.shape[0])
    # for i in range(y2.shape[0]):
    #     temp = (temp * i + square_e[i]) / (i + 1)
    #     mse_list[i] = temp
    # plt.plot(mse_list)
    # plt.title("performance of MSE in LMS")
    # plt.grid()
    # plt.savefig("1.png")
    # plt.show()

    # KLMS
    step_size = 0.9
    kernel_size = 1

    # y3, e3, save_matrix3 = KLMS(input_data, desired, step_size, order, kernel_size,)

    # plt.plot(y3, label='y_pred')
    # plt.plot(desired, label='desired')
    # plt.legend(loc="upper right")
    # plt.xlim(0, 200)
    # plt.title("performance of prediction signal in KLMS")
    # plt.savefig("1.png")
    # plt.show()

    # fig = plt.figure(figsize=(20, 8), dpi=100)
    # plt.subplot(2, 1, 1)
    # plt.specgram(y3, Fs=freq2)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.title("prediction signal")
    # plt.subplot(2, 1, 2)
    # plt.tight_layout()
    # plt.specgram(desired, Fs=freq2)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.title("desired signal")
    # plt.suptitle("frequency spectrogram",fontsize=20)
    # plt.tight_layout()
    # plt.savefig("1.png")
    # plt.show()

    # square_e = e3 ** 2
    # temp = 0
    # mse_list = np.zeros(y3.shape[0])
    # for i in range(y3.shape[0]):
    #     temp = (temp * i + square_e[i]) / (i + 1)
    #     mse_list[i] = temp
    # plt.plot(mse_list)
    # plt.title("performance of MSE in KLMS")
    # plt.grid()
    # plt.savefig("1.png")
    # plt.show()

    # different parameter --klms

    step_size = [0.1, 0.3, 0.6, 0.9]
    kernel_size = [0.01, 0.1, 1, 10]

    # different step size

    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # for step in range(4):
    #     y4, e4, save_matrix4 = KLMS(input_data, desired, step_size[step], order, kernel_size[2])
    #
    #     plt.subplot(4, 1, (step + 1))
    #     plt.plot(y4, label='y_pred，step size={}'.format(step_size[step]))
    #     plt.plot(desired, label='desired')
    #     plt.legend(loc="upper right")
    #     plt.xlim(0, 200)
    #
    #     square_e = e4 ** 2
    #     temp = 0
    #     mse_list = np.zeros(y4.shape[0])
    #     for i in range(y4.shape[0]):
    #         temp = (temp * i + square_e[i]) / (i + 1)
    #         mse_list[i] = temp
    #     plt.plot(mse_list, label="step_size={}".format(step_size[step]))
    #     plt.legend()
    # plt.title("performance of MSE in different step size in KLMS with the kernel size ={}".format(kernel_size[2]))
    # plt.suptitle("performance of prediction signal in different step size in KLMS with the kernel size ={}".format(kernel_size[2]))
    # plt.savefig("1.png")
    # plt.show()

    # different kernel size
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # for step in range(4):
    #     y5, e5, save_matrix5 = KLMS(input_data, desired, step_size[3], order, kernel_size[step])
    #
    #     plt.subplot(4, 1, (step + 1))
    #     plt.plot(y5, label='y_pred，kernel size={}'.format(kernel_size[step]))
    #     plt.plot(desired, label='desired')
    #     plt.legend(loc="upper right")
    #     plt.xlim(0, 200)
    #
    #     square_e = e5 ** 2
    #     temp = 0
    #     mse_list = np.zeros(y5.shape[0])
    #     for i in range(y5.shape[0]):
    #         temp = (temp * i + square_e[i]) / (i + 1)
    #         mse_list[i] = temp
    #     plt.plot(mse_list, label="kernel_size={}".format(kernel_size[step]))
    #     plt.legend()
    # plt.title("performance of MSE in different kernelsize size in KLMS with the step size ={}".format(step_size[3]))
    # plt.suptitle("performance of prediction signal in different step size in KLMS with the step size ={}".format(step_size[3]))
    # plt.savefig("1.png")
    # plt.show()

    # different input freq
    # y6, e6, save_matrix6 = KLMS(input_data, desired, step_size[3], order, kernel_size[2])
    # freq4 = [500, 1000, 1500, 2000]
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    #
    # for step in range(4):
    #     input_data = np.sin(2 * np.pi * freq4[step] * t)
    #     y6 = prediction(input_data, save_matrix6, e6, step_size[3], order, kernel_size[2])
    #     plt.subplot(4, 1, (step + 1))
    #     plt.plot(y6, label='y_pred,input freq = {}'.format(freq4[step]))
    #     plt.plot(desired, label='desired')
    #     plt.legend(loc="upper right")
    #     plt.xlim(0, 200)
    #
    # plt.suptitle("performance of prediction signal in different input data frequency in KLMS ")
    # plt.savefig("1.png")
    # plt.show()

    # new data --klms
    mix = np.random.binomial(1, 0.9, t.shape[0])
    noise = np.zeros(t.shape[0])
    for i in range(t.shape[0]):
        noise[i] = mix[i] * np.random.normal(0, 0.1) + (1 - mix[i]) * np.random.normal(4, 0.1)

    desired = noise + desired

    # plt.plot(t, input_data, label="input")
    # plt.plot(t, desired, label="new desired")
    # plt.xlim(0, 0.01)
    # plt.legend(loc="upper right")
    # plt.title("input data and new desired data in [0,0.01]s")
    # plt.savefig("1.png")
    # plt.show()

    # KLMS-MEE2
    order = 10
    step_size = [0.1, 0.4, 0.7, 0.9]
    # kernel_size for RKHS, kernel_size2 for kernel MEE
    kernel_size = 0.1
    kernel_size2 = [0.01, 0.1, 1, 10]
    window_size = 5
    length_term = 2000


    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # for step in range(4):
    #     y8, e8,save_matrix8 = KLMS_MEE2(input_data, desired, step_size[step], order, kernel_size, kernel_size2[2], length_term)
    #
    #     plt.subplot(4, 1, (step + 1))
    #     plt.plot(y8, label='y_pred，step size={}'.format(step_size[step]))
    #     plt.plot(desired, label='desired')
    #     plt.legend(loc="upper right")
    #     plt.title("learning rate = {:0.2e}".format(step_size[step]/(window_size / kernel_size2[2] ** 2)))
    #     plt.tight_layout()
    #     plt.xlim(0, 1000)
    #
    #     square_e = e8 ** 2
    #     temp = 0
    #     mse_list = np.zeros(length_term)
    #     for i in range(length_term):
    #         temp = (temp * i + square_e[i]) / (i + 1)
    #         mse_list[i] = temp
    #     plt.plot(mse_list, label="kernel_size={}".format(kernel_size2[step]))
    #     plt.legend()
    # plt.title("performance of MSE in different kernel size in MEE in KLMS with the step size ={:0.2e}".format(step_size))
    # plt.suptitle("performance of prediction signal in different kernel size in KLMS with the kernel size in MEE = {}".
    #              format(kernel_size2))
    # plt.tight_layout()
    # plt.savefig("1.png")
    # plt.show()

    # y8, e8,save_matrix8 = KLMS_MEE2(input_data, desired, step_size[1], order, kernel_size, kernel_size2[2], length_term)
    #
    # # performance of prediction signal in KLMS with MEE
    #
    # plt.plot(y8, label='y_pred')
    # plt.plot(desired, label='desired')
    # plt.legend(loc="upper right")
    # plt.xlim(0, 500)
    # plt.title("performance of prediction signal in KLMS with MEE")
    # plt.savefig("1.png")
    # plt.show()

    # square_e = e8 ** 2
    # temp = 0
    # mse_list = np.zeros(length_term)
    # for i in range(length_term):
    #     temp = (temp * i + square_e[i]) / (i + 1)
    #     mse_list[i] = temp
    # plt.plot(mse_list)
    # plt.title("performance of MEE in KLMS")
    # plt.grid()
    # plt.savefig("1.png")
    # plt.show()

    # MSE -- KLMS -- new data
    step_size = 0.9
    kernel_size = 1

    # y3, e3, save_matrix3 = KLMS(input_data, desired, step_size, order, kernel_size)

    # plt.plot(y3, label='y_pred')
    # plt.plot(desired, label='desired')
    # plt.legend(loc="upper right")
    # plt.xlim(0, 200)
    # plt.title("performance of prediction signal in KLMS")
    # plt.savefig("1.png")
    # plt.show()

    # fig = plt.figure(figsize=(20, 8), dpi=100)
    # plt.subplot(2, 1, 1)
    # plt.specgram(y3, Fs=freq2)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.title("prediction signal")
    # plt.subplot(2, 1, 2)
    # plt.tight_layout()
    # plt.specgram(desired, Fs=freq2)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.title("desired signal")
    # plt.suptitle("frequency spectrogram",fontsize=20)
    # plt.tight_layout()
    # plt.savefig("1.png")
    # plt.show()

    # square_e = e3 ** 2
    # temp = 0
    # mse_list = np.zeros(y3.shape[0])
    # for i in range(y3.shape[0]):
    #     temp = (temp * i + square_e[i]) / (i + 1)
    #     mse_list[i] = temp
    # plt.plot(mse_list)
    # plt.title("performance of MSE in KLMS")
    # plt.grid()
    # plt.savefig("1.png")
    # plt.show()

    # histogram of error

    # order = 10
    # step_size = [0.1, 0.4, 0.7, 0.9]
    # # kernel_size for RKHS, kernel_size2 for kernel MEE
    # kernel_size = 0.1
    # kernel_size2 = [0.01, 0.1, 1, 10]
    # window_size = 5
    # length_term = 1500
    #
    # y8, e8, save_matrix8 = KLMS_MEE2(input_data, desired, step_size[1], order, kernel_size, kernel_size2[2], length_term)
    #
    # plt.hist((e8[300:length_term]), density=True, bins=100, label="KLMS_MEE")
    #
    # step_size = 0.9
    # kernel_size = 1
    #
    # y3, e3, save_matrix3 = KLMS(input_data, desired, step_size, order, kernel_size)
    #
    # plt.hist((e3), density=True, bins=100, label="KLMS_MSE")
    # plt.legend()
    # plt.title("distribution histogram of the error in MSE and MEE in KLMS")
    # plt.savefig("1.png")
    # plt.show()



    # different input freq
    order = 10
    step_size = [0.1, 0.4, 0.7, 0.9]
    # kernel_size for RKHS, kernel_size2 for kernel MEE
    kernel_size = 0.1
    kernel_size2 = [0.01, 0.1, 1, 10]
    window_size = 5
    length_term = 500
    y8, e8, save_matrix8 = KLMS_MEE2(input_data, desired, step_size[1], order, kernel_size, kernel_size2[2], length_term)
    freq4 = [500, 1000, 1500, 2000]
    fig = plt.figure(figsize=(16, 9), dpi=100)
    #
    for step in range(4):
        input_data = np.sin(2 * np.pi * freq4[step] * t)
        y6 = prediction2(input_data,save_matrix8,e8,step_size[1],order,kernel_size,kernel_size2[2],length_term)
        plt.subplot(4, 1, (step + 1))
        plt.plot(y6, label='y_pred,input freq = {}'.format(freq4[step]))
        plt.plot(desired, label='desired')
        plt.legend(loc="upper right")
        plt.xlim(0, 200)

    plt.suptitle("performance of prediction signal in different input data frequency in KLMS-MEE ")
    plt.savefig("1.png")
    plt.show()
