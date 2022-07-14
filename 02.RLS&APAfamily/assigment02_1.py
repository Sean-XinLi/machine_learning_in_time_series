import numpy as np
from scipy import signal, stats, linalg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class GenData(object):
    def __init__(self, size, noise_power):
        self.size = size
        # 噪音的power 即 方差
        self.noisy_power = noise_power

    def get_input_data(self):
        input_data = stats.levy_stable.rvs(1.8, 0, 0, 1, size=self.size)
        return input_data

    def normalize_input_data(self, input_data):
        normalize_factor = np.sqrt(np.sum(input_data ** 2))
        input_data = input_data / normalize_factor
        return input_data

    def get_output_data(self, input_data):
        t_in = np.arange(0, self.size)
        system = ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1], [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1)
        self.t, self.output_data = signal.dlsim(system, input_data, t=t_in)
        np.random.seed(1)
        self.output_data = self.output_data.reshape(self.size)
        self.noisy = np.random.normal(0, self.noisy_power, self.size)
        self.output_data = self.output_data + self.noisy

        return self.output_data


class WienerFilter(object):
    # window_size 为 截取 信号 长度， 用于 P 和 R 的计算
    # (即 根据delay的情况(i与k 或者 tao 的取值情况)对两个信号进行动态截取，然后进行相乘)
    def __init__(self, input_data, output_data, size, window_size, order, start=0) -> object:
        self.input_data = input_data
        self.output_data = output_data
        self.window_size = window_size
        self.order = order
        self.size = size
        self.start = start

    def sampling(self, data, start, size):
        self.samples_data = np.zeros(size)
        self.samples_data = data[start:(start + size)]
        return self.samples_data

    def ACF_OR_CCF(self, input_data, output_data):
        # input_data 和 output_data 分别 代表 相关函数的两个输入的信号
        R = np.zeros(self.order)
        s1 = self.sampling(input_data, self.start, self.window_size)
        # tao = k - i
        # R = E(x(n-i)x(n-k))
        # k 为 order, 即 信号 delay 多少,  0<= k < order
        # i 为 另一个信号的 order, 0 <= i <= k
        for tao in range(self.order):
            start = self.start + tao
            s2 = self.sampling(output_data, start, self.window_size)
            # 下面式子需要，写出信号与每一步步骤进行理解
            R[tao] = (self.window_size - tao) / self.window_size * \
                     np.dot(s1[:(self.window_size - tao)], s2[:self.window_size - tao])
        if input_data is output_data:
            # 如果是自相关函数，生成的 应该是 对角线对称矩阵
            R = linalg.toeplitz(R)
        return R

    def find_w(self):
        R = self.ACF_OR_CCF(self.input_data, self.input_data)
        P = self.ACF_OR_CCF(self.input_data, self.output_data)

        # w = R-1 * P
        R_inverse = np.linalg.inv(R)
        w_star = np.dot(R_inverse, P)

        return w_star

    def get_w_list(self):
        w_list = np.zeros(((self.size - self.order), self.order))
        for self.start in range(self.size - self.window_size):
            w = self.find_w()
            w_list[self.start, :] = w
        return w_list

    def predict_signal(self, w):
        # 在input_data 左侧 添加 与 w 个数 相同的 0，方便直接进行卷积
        input_data = np.pad(self.input_data, (self.order, 0), 'constant')

        y_temp = np.zeros(self.size + 1)
        for i in range(input_data.shape[0] - self.order):
            # 生成的预测值均为下一个时点的预测值，因此最后一位存在越界的情况，需要单独考虑
            # 当 i 行至 最后一位 接下来的预测 由于 不在有输入值(输入信号走到尽头)
            # 因此需要对应的将 w 中的 表示当前的 系数 剔除掉
            if i == input_data.shape[0] - self.order:
                sample_input_data = sample_input_data[1:]
                y_temp[i] = np.dot(w[1:], sample_input_data)

            sample_input_data = self.sampling(input_data, i, self.order)
            sample_input_data = sample_input_data[::-1]
            y_temp[i] = np.dot(w, sample_input_data)
        y_pred = y_temp[1:]
        return y_pred

    def normalized_mse(self, w):
        y_pred = self.predict_signal(w)
        err = (self.output_data - y_pred) ** 2
        e_max = err.max(axis=0)
        e_min = err.min(axis=0)
        for i in range(err.shape[0]):
            err[i] = (err[i] - e_min) / (e_max - e_min)
        err_sum = err.sum(axis=0) / self.output_data.shape[0]
        return err_sum, err

    def wr_error(self, w):
        length = self.size - self.order
        y_temp = np.zeros(length)
        e_temp = np.zeros(length)
        for i in range(length):
            input_seg = self.input_data[i:i + self.order]
            input_seg = input_seg[::-1]
            y_pred = np.dot(w, input_seg)
            e = self.output_data[i + self.order] - y_pred
            y_temp[i] = y_pred
            e_temp[i] = e

        return e_temp

    def LMS(self):
        R = self.ACF_OR_CCF(self.input_data, self.input_data)
        P = self.ACF_OR_CCF(self.input_data, self.output_data)

        lam_max = np.max(np.linalg.eigvals(R))
        lam_min = np.min(np.linalg.eigvals(R))
        learning_rate = 50 / (R.shape[0] * np.sum(self.input_data ** 2))
        length = self.size - self.order

        y_temp = np.zeros(length)
        e_temp = np.zeros(length)
        w_temp = np.zeros((length, self.order))

        for i in range(length):
            if i == 0:
                w = np.random.normal(loc=0, scale=1, size=self.order)

            input_seg = self.input_data[i:i + self.order]
            input_seg = input_seg[::-1]
            y_pred = np.dot(w, input_seg)
            e = self.output_data[i + self.order] - y_pred
            w = w + learning_rate * e * input_seg

            y_temp[i] = y_pred
            e_temp[i] = e
            w_temp[i, :] = w.T

        return w, w_temp, e_temp

    def NLMS(self):

        length = self.size - self.order
        y_temp = np.zeros(length)
        e_temp = np.zeros(length)
        w_temp = np.zeros((length, self.order))
        eta = 0.1

        for i in range(length):
            if i == 0:
                w = np.random.normal(loc=0, scale=1, size=self.order)

            input_seg = self.input_data[i:i + self.order]
            input_seg = input_seg[::-1]
            y_pred = np.dot(w, input_seg)
            e = self.output_data[i + self.order] - y_pred

            w = w + eta / (np.dot(input_seg, input_seg)) * e * input_seg

            y_temp[i] = y_pred
            e_temp[i] = e
            w_temp[i, :] = w.T

        return w, w_temp, e_temp

    def W_S_N_R(self, fun):
        if fun == 1:
            w = self.find_w()
        elif fun == 2:
            w, w_list, e = self.LMS()
        if w.shape[0] <= 10:
            w_star = np.ones(w.shape[0])
        else:
            w_star = np.zeros(w.shape[0])
            w_star[0:10] = 1
        result = 10 * np.log10(np.dot(w_star, w_star) / np.dot((w_star - w), (w_star - w)))
        return result

    def mse(self, w):
        length = self.size - self.order
        e_temp = np.zeros(length)
        for i in range(length):
            input_seg = self.input_data[i:i + self.order]
            input_seg = input_seg[::-1]
            y_pred = np.dot(w, input_seg)
            e = self.output_data[i + self.order] - y_pred
            e_temp[i] = e ** 2
        mse = np.sum(e_temp) / length

        return mse




class RLS(object):
    def __init__(self, input_data, output_data, size, window_size, order, forget_factor) -> object:
        self.input_data = input_data
        self.output_data = output_data
        self.window_size = window_size
        self.order = order
        self.size = size
        self.forget_factor = forget_factor

    def run_rls(self):
        identity_matrix = np.eye(self.order)
        c = 100
        length = self.size - self.order
        y_pred = np.zeros(length)
        e = np.zeros(length)
        w = np.zeros((length, self.order))

        for i in range(length - 1):
            if i == 0:
                p_last = identity_matrix * c
                w_last = np.random.normal(loc=0, scale=1, size=self.order)
            output_seg = self.output_data[i + self.order]
            input_seg = self.input_data[i:i + self.order]
            input_seg = input_seg[::-1]

            K = (input_seg.T @ p_last) / (self.forget_factor + input_seg.T @ p_last @ input_seg)

            y_temp = np.dot(input_seg, w_last)
            e_temp = output_seg - y_temp
            w_temp = w_last + K * e_temp
            p = (identity_matrix - np.dot(K, input_seg)) * p_last / self.forget_factor

            p_last = p
            w_last = w_temp
            y_pred[i] = y_temp
            e[i] = e_temp
            w[i, :] = w_temp.T
        return w_temp, w, e

    def W_S_N_R(self):
        w, w_list, e = self.run_rls()
        if w.shape[0] <= 10:
            w_star = np.ones(w.shape[0])
        else:
            w_star = np.zeros(w.shape[0])
            w_star[0:10] = 1
        result = 10 * np.log10(np.dot(w_star, w_star) / np.dot((w_star - w), (w_star - w)))
        return result

    def mse(self, w):
        length = self.size - self.order
        e_temp = np.zeros(length)
        for i in range(length):
            input_seg = self.input_data[i:i + self.order]
            input_seg = input_seg[::-1]
            y_pred = np.dot(w, input_seg)
            e = self.output_data[i + self.order] - y_pred
            e_temp[i] = e ** 2
        mse = np.sum(e_temp) / length

        return mse


if __name__ == '__main__':
    # 设置参数
    size = 10000
    window_size = 100
    order = 10
    noise_power = 0.1
    forget_factor = 0.9955

    # print('weiner WSNR in different filter order and different noise power')
    # print('order\t\t\tN(power of noise)\t\tWSNR\t\t\t\t\tmse_e')
    # order_list = [5, 10, 15, 30]
    # noise_power_list = [0.1, 0.3, 1.5]
    # for noise_power in noise_power_list:
    #     gen = GenData(size, noise_power)
    #     input_data = gen.get_input_data()
    #     output_data = gen.get_output_data(input_data)
    #     for order in order_list:
    #         wr = WienerFilter(input_data, output_data, size, window_size, order)
    #         w1 = wr.find_w()
    #         w1_list = wr.get_w_list()
    #         res1 = wr.W_S_N_R(1)
    #         e1 = wr.wr_error(w1)
    #         e1 = e1 / np.sqrt(np.sum(input_data ** 2))
    #         mse_e1 = e1 ** 2
    #         sum_mse_e1 = np.sum(mse_e1) / mse_e1.shape[0]
    #         print('{:^} \t\t\t\t{:^} \t\t\t\t\t{:^} \t\t{:^}'.format(order, noise_power, res1, sum_mse_e1))
    #
    # print('LMS WSNR in different filter order and different noise power')
    # print('order\t\t\tN(power of noise)\t\tWSNR\t\t\t\t\tmse_e')
    # order_list = [5, 10, 15, 30]
    # noise_power_list = [0.1, 0.3, 1.5]
    # for noise_power in noise_power_list:
    #     gen = GenData(size, noise_power)
    #     input_data = gen.get_input_data()
    #     input_data = gen.normalize_input_data(input_data)
    #     output_data = gen.get_output_data(input_data)
    #     for order in order_list:
    #         wr = WienerFilter(input_data, output_data, size, window_size, order)
    #         w2, w2_list, e2 = wr.LMS()
    #         mse_e2 = e2 ** 2
    #         print(e2)
    #         sum_mse_e2 = np.sum(mse_e2) / mse_e2.shape[0]
    #         res2 = wr.W_S_N_R(2)
    #         print('{:^} \t\t\t\t{:^} \t\t\t\t\t{:^} \t\t{:^}'.format(order, noise_power, res2, sum_mse_e2))

    # print('RLS WSNR in different filter order and different noise power')
    # print('order\t\t\tN(power of noise)\t\tWSNR\t\t\t\t\tmse_e')
    # order_list = [5, 10,15, 30]
    # noise_power_list = [0.1, 0.3, 1.5]
    # for noise_power in noise_power_list:
    #     gen = GenData(size, noise_power)
    #     input_data = gen.get_input_data()
    #     input_data = gen.normalize_input_data(input_data)
    #     output_data = gen.get_output_data(input_data)
    #     for order in order_list:
    #         r = RLS(input_data, output_data, size, window_size, order, forget_factor)
    #         w3, w3_list, e3 = r.run_rls()
    #         mse_e3 = e3 ** 2
    #         sum_mse_e3 = np.sum(mse_e3) / mse_e3.shape[0]
    #         res3 = r.W_S_N_R()
    #         print('{:^} \t\t\t\t{:^} \t\t\t\t\t{:^} \t\t{:^}'.format(order, noise_power, res3, sum_mse_e3))

    # 设置参数
    size = 10000
    window_size = 100
    order = 5
    noise_power = 0.1
    forget_factor = 0.9955

    # 生成数据
    gen = GenData(size, noise_power)
    input_data = gen.get_input_data()
    output_data = gen.get_output_data(input_data)
    wr = WienerFilter(input_data, output_data, size, window_size, order)
    w1 = wr.find_w()
    w1_list = wr.get_w_list()
    res1 = wr.W_S_N_R(1)
    e1 = wr.wr_error(w1)
    e1 = e1 / np.sqrt(np.sum(input_data ** 2))
    mse_e1 = e1 ** 2
    sum_mse_e1 = np.sum(mse_e1) / mse_e1.shape[0]

    input_data = gen.normalize_input_data(input_data)
    output_data = gen.get_output_data(input_data)
    wr = WienerFilter(input_data, output_data, size, window_size, order)
    w2, w2_list, e2 = wr.LMS()
    mse_e2 = e2 ** 2
    sum_mse_e2 = np.sum(mse_e2) / mse_e2.shape[0]
    res2 = wr.W_S_N_R(2)

    r = RLS(input_data, output_data, size, window_size, order, forget_factor)
    w3, w3_list, e3 = r.run_rls()
    mse_e3 = e3 ** 2
    sum_mse_e3 = np.sum(mse_e3) / mse_e3.shape[0]
    res3 = r.W_S_N_R()

    # test - NLMS
    w2, w2_list, e2 = wr.NLMS()
    print(w2_list)

    # mse1_list = []
    # mse2_list = []
    # mse3_list = []
    # for i in range(w1_list.shape[0]):
    #     print(i)
    #     mse1 = wr.mse(w1_list[i, :])
    #     mse2 = wr.mse(w2_list[i, :])
    #     mse3 = r.mse(w3_list[i, :])
    #     mse1_list.append(mse1)
    #     mse2_list.append(mse2)
    #     mse3_list.append(mse3)
    #
    # # mse
    # x = np.arange(order, (order + mse_e1.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # ax1 = plt.subplot(3, 1, 1)
    # ax1 = plt.plot(x, mse1_list, c='red', label='wiener solution')
    # plt.legend(loc='upper right')
    # ax2 = plt.subplot(3, 1, 2)
    # ax2 = plt.plot(x, mse2_list, c='blue', label='LMS')
    # plt.legend(loc='upper right')
    # ax3 = plt.subplot(3, 1, 3)
    # ax3 = plt.plot(x, mse3_list, c='green', label='RLS')
    # plt.legend(loc='upper right')
    # plt.suptitle('performance of converging speed in order={}'.format(order))
    # plt.show()

    # help
    # x = np.arange(order, (order + mse_e1.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # plt.plot(x, mse1_list)
    # plt.title('wiener solution, window size = {}, filter order = {}'.format(window_size, order))
    # plt.show()
    #
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # plt.plot(x, mse2_list)
    # plt.title('LMS, filter order = {}'.format(order))
    # plt.show()
    #
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # plt.plot(x, mse3_list)
    # plt.title('RLS, filter order = {}'.format(order))
    # plt.show()


    # w
    # x = np.arange(order, (order + mse_e1.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # ax1 = plt.subplot(3, 1, 1)
    # for i in range(order):
    #     ax1 = plt.plot(x, w1_list[:, i])
    # ax1 = plt.title("wiener solution")
    # ax2 = plt.subplot(3, 1, 2)
    # for i in range(order):
    #     ax2 = plt.plot(x, w2_list[:, i])
    # ax2 = plt.title("LMS")
    # ax3 = plt.subplot(3, 1, 3)
    # for i in range(order):
    #     ax3 = plt.plot(x, w3_list[:, i])
    # ax2 = plt.title("RLS")
    # plt.suptitle("converging performance in weights(filter order= {})".format(order))
    # plt.show()

    # mse
    # x = np.arange(order, (order + mse_e1.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # ax1 = plt.subplot(3, 1, 1)
    # ax1 = plt.plot(x, mse_e1, c='red', label='wiener solution')
    # plt.ylim((0, 0.3))
    # plt.legend(loc='best')
    # ax2 = plt.subplot(3, 1, 2)
    # ax2 = plt.plot(x, mse_e2, c='blue', label='LMS')
    # plt.ylim((0, 0.3))
    # plt.legend(loc='best')
    # ax3 = plt.subplot(3, 1, 3)
    # ax3 = plt.plot(x, mse_e3, c='green', label='RLS')
    # plt.legend(loc='best')
    # plt.ylim((0, 0.3))
    # plt.show()