import os
import warnings
import wave
import numpy as np
from scipy import signal, stats, linalg
from scipy.io import wavfile
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('error')


class WinenerFilter_Non_Stationary(object):
    def __init__(self, data, size, window_size, order, start, iteration=100):
        self.input_data = data
        self.output_data = data
        self.window_size = window_size
        self.order = order
        self.size = size
        self.start = start
        self.iteration = iteration

    def sampling(self, data, start, size):
        self.samples_data = np.zeros(size)
        self.samples_data = data[start:(start + size)]
        # 如果sample后，大小不足size，后面补0
        if self.samples_data.shape[0] != size:
            self.samples_data = np.pad(self.samples_data, (0, (size - self.samples_data.shape[0])), 'constant')

        return self.samples_data

    def ACF_OR_CCF(self, input_data, output_data, start, ch):
        # input_data 和 output_data 分别 代表 相关函数的两个输入的信号
        R = np.zeros(self.order)
        s1 = self.sampling(input_data, start, self.window_size)
        # tao = k - i
        # R = E(x(n-i)x(n-k))
        # k 为 order, 即 信号 delay 多少,  0<= k < order
        # i 为 另一个信号的 order, 0 <= i <= k
        for tao in range(self.order):
            point = start + tao
            s2 = self.sampling(output_data, point, self.window_size)
            # 下面式子需要，写出信号与每一步步骤进行理解
            v1 = s1[:(self.window_size - tao)]
            v2 = s2[:self.window_size - tao]
            # if v1.shape[0] != v2.shape[0]:
            #     v1 = v1[:(v1.shape[0]-1)]
            R[tao] = (self.window_size - tao) / self.window_size * np.dot(v1, v2)

        if ch == 1:
            # 如果是自相关函数，生成的 应该是 对角线对称矩阵
            R = linalg.toeplitz(R)
        elif ch == 2:
            return R
        return R

    def wiener_solution(self, start):
        R = self.ACF_OR_CCF(self.input_data, self.input_data, start, ch=1)
        P = self.ACF_OR_CCF(self.input_data, self.output_data, start, ch=2)
        # w = R-1 * P
        R_inverse = np.linalg.inv(R)
        w_star = np.dot(R_inverse, P)
        return w_star

    def normalized_mse(self, y_pred):
        err = (self.output_data - y_pred) ** 2
        e_max = err.max(axis=0)
        e_min = err.min(axis=0)
        for i in range(err.shape[0]):
            err[i] = (err[i] - e_min) / (e_max - e_min)
        return err

    def LMS(self, start):
        R = self.ACF_OR_CCF(self.input_data, self.input_data, start, ch=1)
        P = self.ACF_OR_CCF(self.input_data, self.output_data, start, ch=2)
        # 学习率: 0 < learning rate < (1 / (R 对应的最大的特征值))
        lam_max = np.max(np.linalg.eigvals(R))
        lam_min = np.min(np.linalg.eigvals(R))
        learning_rate = 1 / 2*(lam_max + lam_min)
        i_matrix = np.identity(R.shape[0])
        temp = i_matrix - learning_rate * R
        w = np.zeros(self.order)
        # LMS ：w_(m+1) = learning_rate * ( I - learning_rate * R) ** m * P
        # 0 <= m < iteration

        i = 0
        # e = [10**10, 10**9]
        while i < self.iteration:
        # while i < self.iteration and e[-1] < e[-2]:
            w_temp = w
            w = np.dot(temp, w_temp) + learning_rate * P
            b = w
            sample_input_data = self.input_data[start:(start + self.order)]
            sample_input_data = sample_input_data[::-1]
            y_pred = np.dot(w, sample_input_data)
            a = self.output_data[start + self.order]
            # try:
            #     a = self.output_data[start + self.order]
            #     e_temp = (y_pred - self.output_data[start + self.order]) ** 2
            #     e.append(e_temp)
            # except RuntimeWarning as warn:
            #     e.append(0)
            i += 1
        if i == self.iteration:
            return w
        else:
            return w_temp

    def predict_non_stationary(self, method):
        # 在input_data 左侧 添加 与 w 个数 相同的 0，方便直接进行卷积
        # input_data = np.pad(self.input_data, (self.order, 0), 'constant')
        y_temp = np.zeros(self.size - self.order)
        for start in range(self.start, (self.output_data.shape[0] - self.order)):
            if method == 1:
                w = self.wiener_solution(start)
            elif method == 2:
                w = self.LMS(start)
            sample_input_data = self.input_data[start:(start + self.order)]
            sample_input_data = sample_input_data[::-1]
            y_temp[start] = np.dot(w, sample_input_data)
        y_pred = np.pad(y_temp, (self.order, 0), 'constant')
        y_pred[:self.order] = self.output_data[:self.order]
        return y_pred


if __name__ == '__main__':
    # open a audio file
    file = "/speech.WAV"
    path = os.getcwd()
    file_path = path + file
    software = "open -a /Applications/IINA.app"
    # os.system(software + " " + file_path)

    # 读取参数
    f = wave.open(file_path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print(params)

    # 获得信号
    # 16进制的数据
    strData = f.readframes(nframes)
    f.close()

    # 转换成十进制
    waveData = np.frombuffer(strData, dtype=np.int16)
    # print(waveData.shape)

    size = waveData.shape[0]
    window_size = 100
    order = 6
    start = 0
    wf = WinenerFilter_Non_Stationary(waveData, size, window_size, order, start)
    y_pred = wf.predict_non_stationary(method=2)
    err = wf.normalized_mse(y_pred)
    # for i, ele in enumerate(err):
    #     print("i:{} --ele:{}".format(i,ele))
    # err_sum = np.sum(err, axis=0) / err.shape[0]
    # print(err_sum)
    # print(y_pred.shape)
    print(y_pred[6:20])
    print(waveData[6:20])

    file = "/speech_pred.WAV"
    file_path = path + file
    # save
    # wavfile.write(file_path, framerate, y_pred.astype(np.int16))
    # os.system(software + " " + file_path)

    # fig = plt.figure(figsize=(20, 8), dpi=100)
    # f, t, Sxx = signal.spectrogram(strData, framerate)
    # plt.pcolormesh(t, f, Sxx, cmap="summer")
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.1, vmax=0.3))
    # sm.set_array

    # plt.show()
