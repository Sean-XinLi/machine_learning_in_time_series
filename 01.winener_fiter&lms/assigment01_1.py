import numpy as np
from scipy import signal, stats, linalg


class GenData(object):
    def __init__(self, size, noise_power):
        self.size = size
        # 噪音的power 即 方差
        self.noisy_power = noise_power

    def get_input_data(self):
        self.input_data = stats.levy_stable.rvs(1.8, 0, 0, 1, size=self.size)
        return self.input_data

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
    def __init__(self, input_data, output_data, size, window_size, order, start, iteration) -> object:
        self.input_data = input_data
        self.output_data = output_data
        self.window_size = window_size
        self.order = order
        self.size = size
        self.start = start
        self.iteration = iteration

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

    def predict_signal(self, w):
        # 在input_data 左侧 添加 与 w 个数 相同的 0，方便直接进行卷积
        input_data = np.pad(self.input_data, (self.order, 0), 'constant')

        y_temp = np.zeros(self.size+1)
        for i in range(input_data.shape[0]-self.order):
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

    def LMS(self):
        R = self.ACF_OR_CCF(self.input_data, self.input_data)
        P = self.ACF_OR_CCF(self.input_data, self.output_data)

        lam_max = np.max(np.linalg.eigvals(R))
        lam_min = np.min(np.linalg.eigvals(R))
        learning_rate = 0.05 / np.sum(np.linalg.eigvals(R))
        learning_rate =  0.1 * R.shape[0] * np.sum(self.input_data ** 2)

        length = self.size - self.order - 1

        y_temp = np.zeros(length)
        e_temp = np.zeros(length)
        w_temp = np.zeros((length, self.order))
        for i in range(length-1):
            if i == 0:
                w = np.zeros(self.order)

            input_seg = self.input_data[i:i+self.order]
            input_seg = input_seg[::-1]
            y_pred = np.dot(w, input_seg)
            e = self.output_data[i+self.order]
            w = w + learning_rate * e * input_seg

            y_temp[i] = y_pred
            e_temp[i] = e
            w_temp[i, :] = w.T

        return w

    def W_S_N_R(self, fun):
        if fun == 1:
            w = self.find_w()
        elif fun == 2:
            w = self.LMS()
        w_star = np.ones(self.order)
        result = 10 * np.log10(np.dot(w.T, w) / np.dot((w_star - w).T, (w_star - w)))
        return result


if __name__ == '__main__':
    size = 10000
    window_size = 100
    order = 10
    start = 0
    noise_power = 0.1
    iteration = 300
    gen = GenData(size, noise_power)

    input_data = gen.get_input_data()
    input_data = gen.normalize_input_data(input_data)
    output_data = gen.get_output_data(input_data)

    wf = WienerFilter(input_data, output_data, size, window_size, order, start, iteration)
    w = wf.find_w()
    print(w)
    w = wf.LMS()
    print(w)
    # y_pred = wf.predict_signal(w)
    # print(y_pred)
    # err_sum, err = wf.normalized_mse(w)
    # print(err_sum)
    # w2 = wf.LMS()
    # print(w2)
    # res = wf.W_S_N_R(1)
    # print(res)
    # res = wf.W_S_N_R(2)
    # print(res)
    # print(err_sum)
