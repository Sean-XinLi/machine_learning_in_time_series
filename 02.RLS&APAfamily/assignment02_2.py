import os
import warnings
import wave
import numpy as np
import scipy.io
from scipy import signal, stats, linalg
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time


class APA_Non_Stationary(object):
    def __init__(self, data, size, filter_length, order, learning_rate, lamda):
        #  获得数据
        normalize_factor = np.sqrt(np.sum(data ** 2))
        input_data = data / normalize_factor
        self.input_data = input_data
        self.output_data = input_data
        self.filter_length = filter_length
        self.order = order
        self.size = size
        self.learning_rate = learning_rate
        self.lamda = lamda

        # 初始化变量
        self.y_pred = np.zeros(self.size)
        self.w = np.zeros((self.order, self.size))
        self.e = np.zeros(self.size)
        self.X = np.zeros((self.order, self.filter_length))
        self.w[:, 1] = np.random.normal(loc=0, scale=1, size=self.order)

    def run_apa(self):
        for step in range((self.filter_length + self.order - 2), self.size):
            for i in range(self.order):
                for j in range(self.filter_length):
                    self.X[i, j] = self.input_data[step - self.filter_length - self.order + i + j + 2]

            self.y_pred[(step - self.filter_length + 1):(step + 1)] = self.X.T @ self.w[:, (step - self.filter_length)]
            e_temp = self.output_data[(step - self.filter_length + 1):(step + 1)] \
                     - self.y_pred[(step - self.filter_length + 1):(step + 1)]
            R_temp = np.linalg.inv(self.X.T @ self.X + self.lamda * np.eye(self.filter_length))
            self.w[:, (step - self.filter_length + 1)] = (1 - self.learning_rate) * \
                                                         self.w[:, (step - self.filter_length)] + \
                                                         self.learning_rate * self.X @ R_temp @ \
                                                         self.output_data[(step - self.filter_length + 1):(step + 1)]
            self.e[(step - self.filter_length + 1):(step + 1)] = e_temp
        length = step - self.filter_length + 1
        while length < self.size - 1:
            self.w[:, length + 1] = self.w[:, step - self.filter_length+1]
            length = length + 1
        return self.w, self.e, self.y_pred

    def mse(self, w):
        length = self.size - self.order
        e_temp = np.zeros(length)
        for i in range(length):
            input_seg = self.input_data[i:i + self.order]
            input_seg = input_seg[::-1]
            y_pred = np.dot(w, input_seg)
            e = self.output_data[i + self.order] - y_pred
            e_temp[i] = e ** 2
        result = np.sum(e_temp) / length

        return result


class RLS(object):
    def __init__(self, input_data, size, order, forget_factor) -> object:
        normalize_factor = np.sqrt(np.sum(input_data ** 2))
        input_data = input_data / normalize_factor
        self.input_data = input_data
        self.output_data = input_data
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

        for i in range(length):
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
        return w, e, y_pred

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

            w = w + eta / (10 ** 3 + np.dot(input_seg, input_seg)) * e * input_seg

            y_temp[i] = y_pred
            e_temp[i] = e
            w_temp[i, :] = w.T

        return w_temp, e_temp, y_temp


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
    lamda = 0.9
    order = 6
    learning_rate = 0.0008
    filter_length = 20

    # # 测试NLMS
    # rls = RLS(waveData, size, order, forget_factor=(1 - learning_rate))
    # w1, e1, y_pred1 = rls.NLMS()
    # y_pred1 = y_pred1 * np.sqrt(np.sum(waveData ** 2))
    # file = "/speech_pred_NLMS.WAV"
    # file_path = path + file
    # wavfile.write(file_path, framerate, y_pred1.astype(np.int16))
    # os.system(software + " " + file_path)

    # RLS
    # rls = RLS(waveData, size, order, forget_factor=(1 - learning_rate))
    # w1, e1, y_pred1 = rls.run_rls()
    # mse_e1 = e1 ** 2
    # sum_mse1 = np.sum(mse_e1)
    # y_pred1 = y_pred1 * np.sqrt(np.sum(waveData ** 2))
    # x = np.arange(order, (order + y_pred1.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # for i in range(order):
    #     plt.plot(x, w1[:, i])
    #     plt.title("RLS filter parameters change over time(filter order={})".format(order), fontsize=18)
    #
    # plt.show()

    # x = np.arange(order, (order + y_pred1.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # plt.plot(x, e1)
    # plt.title("RLS error between predicted signal and ture singal change over time(filter order={})".format(order), fontsize=18)
    # plt.show()

    # save
    # file = "/speech_pred_rls_15.WAV"
    # file_path = path + file
    # wavfile.write(file_path, framerate, y_pred1.astype(np.int16))
    # os.system(software + " " + file_path)

    # mse1_list = []
    # for i in range(w1.shape[0]):
    #     print(i)
    #     mse1 = rls.mse(w1[i, :])
    #     mse1_list.append(mse1)
    # # mse
    # x = np.arange(order, (order + mse_e1.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # plt.plot(x, mse1_list)
    # plt.suptitle('performance of converging speed in order={}'.format(order))
    # plt.show()

    # 频谱图
    # fig = plt.figure(figsize=(20, 14), dpi=100)
    # ax1 = plt.subplot(2, 1, 1)
    # ax1 = plt.specgram(y_pred1, Fs=nframes)
    # plt.xlabel('Time')
    # plt.ylabel('frequency')
    # plt.title('RLS')
    # ax2 = plt.subplot(2, 1, 2)
    # ax2 = plt.specgram(waveData, Fs=nframes)
    # plt.xlabel('Time')
    # plt.ylabel('frequency')
    # plt.title('origin data')
    # plt.suptitle('spectrogram (filter order={})'.format(order), fontsize=18)
    # plt.show()

    # forget_factor -- w
    # step_list = np.arange(1, 17)
    # fig = plt.figure(figsize=(20, 14), dpi=100)
    # for step in step_list:
    #     learning_rate = round(step * 0.0003, 4)
    #     forget_factor = 1 - learning_rate
    #     rls = RLS(waveData, size, order, forget_factor)
    #     w, e, y_pred = rls.run_rls()
    #     x = np.arange(order, (order + y_pred.shape[0]))
    #     plt.subplot(4, 4, (step - (15 * (step_list[0] // 15))))
    #     for i in range(order):
    #         plt.plot(x, w[:, i])
    #     ax = plt.title("forget_factor={}".format(forget_factor))
    # plt.suptitle("weights performance in different forget factor (filter order={})".format(order), fontsize=18)
    # plt.show()
    #
    # forget_factor -- mse
    # x = np.arange(order, (order + y_pred1.shape[0]))
    # forget_factor_list = []
    # sum_mse_list = []
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # step_list = np.arange(1, 16)
    # for step in step_list:
    #     learning_rate = round(step * 0.0003, 4)
    #     forget_factor = 1 - learning_rate
    #     rls = RLS(waveData, size, order, forget_factor)
    #     w, e, y_pred = rls.run_rls()
    #     mse_e = e ** 2
    #     sum_mse = np.sum(mse_e) / mse_e.shape[0]
    #     sum_mse_list.append(sum_mse)
    #     forget_factor_list.append(forget_factor)
    #     plt.subplot(4, 4, (step - (15 * (step_list[0] // 15))))
    #     plt.plot(x, mse_e, label='forget_factor={}'.format(forget_factor))
    #     plt.legend(loc='upper right')
    # plt.suptitle("The MSE performance in different forget factor (filter order={})".format(order), fontsize=18)
    # plt.show()
    #
    # step_list = np.arange(16, 31)
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # for step in step_list:
    #     learning_rate = round(step * 0.0003, 4)
    #     forget_factor = 1 - learning_rate
    #     rls = RLS(waveData, size, order, forget_factor)
    #     w, e, y_pred = rls.run_rls()
    #     mse_e = e ** 2
    #     sum_mse = np.sum(mse_e) / mse_e.shape[0]
    #     sum_mse_list.append(sum_mse)
    #     forget_factor_list.append(forget_factor)
    #
    #     plt.subplot(4, 4, (step - (15 * (step_list[0] // 15))))
    #     plt.plot(x, mse_e, label='forget_factor={}'.format(forget_factor))
    #     plt.legend(loc='upper right')
    # plt.suptitle("The MSE performance in different forget factor (filter order={})".format(order), fontsize=18)
    # plt.show()
    #
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # plt.plot(forget_factor_list, sum_mse_list)
    # plt.suptitle("The relationship between MSE and forget factor (filter order={})".format(order), fontsize=18)
    # plt.show()

    # APA
    # apa = APA_Non_Stationary(waveData, size, filter_length, order, learning_rate, lamda)
    # w, e, y_pred = apa.run_apa()
    # y_pred = y_pred * np.sqrt(np.sum(waveData ** 2))
    # mse_e = e ** 2
    # sum_mse = np.sum(mse_e)

    # w -- best parameter
    # x = np.arange(order, (order + y_pred.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # for i in range(order):
    #     plt.plot(x, w.T[:, i])
    #     plt.title("APA filter parameters change over time(filter order={})".format(order), fontsize=18)
    # plt.show()

    # save
    # file = "/speech_pred_apa_15.WAV"
    # file_path = path + file
    # wavfile.write(file_path, framerate, y_pred.astype(np.int16))
    # os.system(software + " " + file_path)

    # 频谱图
    # fig = plt.figure(figsize=(20, 14), dpi=100)
    # ax1 = plt.subplot(3, 1, 1)
    # ax1 = plt.specgram(y_pred, Fs=nframes)
    # plt.xlabel('Time')
    # plt.ylabel('frequency')
    # plt.title('APA')
    # ax2 = plt.subplot(3, 1, 2)
    # ax2 = plt.specgram(y_pred1, Fs=nframes)
    # plt.xlabel('Time')
    # plt.ylabel('frequency')
    # plt.title('RLS')
    # ax3 = plt.subplot(3, 1, 3)
    # ax3 = plt.specgram(waveData, Fs=nframes)
    # plt.xlabel('Time')
    # plt.ylabel('frequency')
    # plt.title('origin data')
    # plt.suptitle('spectrogram (filter order={})'.format(order), fontsize=18)
    # plt.show()

    # x = np.arange(order, (order + y_pred.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # plt.plot(x, e)
    # plt.title("APA error between predicted signal and ture singal change over time(filter order={})".format(order), fontsize=18)
    # plt.show()

    # filter_length -- mse
    # x = np.arange(order, (order + y_pred.shape[0]))
    # filter_length_list = []
    # sum_mse_list = []
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # step_list = np.arange(1, 16)
    # for step in step_list:
    #     filter_length = step
    #     apa = APA_Non_Stationary(waveData, size, filter_length, order, learning_rate, lamda)
    #     w, e, y_pred = apa.run_apa()
    #     mse_e = e ** 2
    #     sum_mse = np.sum(mse_e) / mse_e.shape[0]
    #     sum_mse_list.append(sum_mse)
    #     filter_length_list.append(filter_length)
    #     plt.subplot(4, 4, (step-(15 * (step_list[0] // 15))))
    #     plt.plot(x, mse_e, label='filter_length={}'.format(filter_length))
    #     plt.legend(loc='upper right')
    # plt.show()
    #
    # step_list = np.arange(16, 31)
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # for step in step_list:
    #     filter_length = step
    #     apa = APA_Non_Stationary(waveData, size, filter_length, order, learning_rate, lamda)
    #     w, e, y_pred = apa.run_apa()
    #     mse_e = e ** 2
    #     sum_mse = np.sum(mse_e) / mse_e.shape[0]
    #     sum_mse_list.append(sum_mse)
    #     filter_length_list.append(filter_length)
    #     plt.subplot(4, 4, (step-(15 * (step_list[0] // 15))))
    #     plt.plot(x, mse_e, label='filter_length={}'.format(filter_length))
    #     plt.legend(loc='upper right')
    # plt.show()
    #
    # step_list = np.arange(31, 46)
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # for step in step_list:
    #     filter_length = step
    #     apa = APA_Non_Stationary(waveData, size, filter_length, order, learning_rate, lamda)
    #     w, e, y_pred = apa.run_apa()
    #     mse_e = e ** 2
    #     sum_mse = np.sum(mse_e) / mse_e.shape[0]
    #     sum_mse_list.append(sum_mse)
    #     filter_length_list.append(filter_length)
    #     plt.subplot(4, 4, (step - (15 * (step_list[0] // 15))))
    #     plt.plot(x, mse_e, label='filter_length={}'.format(filter_length))
    #     plt.legend(loc='upper right')
    # plt.show()
    #
    # step_list = np.arange(46, 61)
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # for step in step_list:
    #     filter_length = step
    #     apa = APA_Non_Stationary(waveData, size, filter_length, order, learning_rate, lamda)
    #     w, e, y_pred = apa.run_apa()
    #     mse_e = e ** 2
    #     sum_mse = np.sum(mse_e) / mse_e.shape[0]
    #     sum_mse_list.append(sum_mse)
    #     filter_length_list.append(filter_length)
    #     plt.subplot(4, 4, (step - (15 * (step_list[0] // 15))))
    #     plt.plot(x, mse_e, label='filter_length={}'.format(filter_length))
    #     plt.legend(loc='upper right')
    # plt.show()
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # plt.plot(filter_length_list, sum_mse_list)
    # plt.suptitle("The relationship between MSE and filter_length (filter order={})".format(order), fontsize=18)
    # plt.show()

    # filter_length -- w
    # x = np.arange(order, (order + y_pred.shape[0]))
    # fig = plt.figure(figsize=(20, 14), dpi=100)
    # step_list = np.arange(1, 16)
    # for step in step_list:
    #     filter_length = step
    #     apa = APA_Non_Stationary(waveData, size, filter_length, order, learning_rate, lamda)
    #     w, e, y_pred = apa.run_apa()
    #     plt.subplot(4, 4, (step - (15 * (step_list[0] // 15))))
    #     for i in range(order):
    #         plt.plot(x, w.T[:, i])
    #     ax = plt.title("filter_length={}".format(filter_length))
    # plt.suptitle("the weights performance in different filter length(filter order={})".format(order), fontsize=18)
    # plt.show()

    # forget_factor -- w
    # step_list = np.arange(1, 17)
    # fig = plt.figure(figsize=(20, 14), dpi=100)
    # for step in step_list:
    #     learning_rate = round(step * 0.0003, 4)
    #     forget_factor = 1 - learning_rate
    #     apa = APA_Non_Stationary(waveData, size, filter_length, order, learning_rate, lamda)
    #     w, e, y_pred = apa.run_apa()
    #     x = np.arange(order, (order + y_pred.shape[0]))
    #     plt.subplot(4, 4, (step - (15 * (step_list[0] // 15))))
    #     for i in range(order):
    #         plt.plot(x, w.T[:, i])
    #     ax = plt.title("forget_factor={}".format(forget_factor))
    # plt.suptitle("the weights performance in different forget factor(filter order={})".format(order), fontsize=18)
    #
    # plt.show()

    # forget_factor -- mse
    # x = np.arange(order, (order + y_pred.shape[0]))
    # forget_factor_list = []
    # sum_mse_list = []
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # step_list = np.arange(1, 16)
    # for step in step_list:
    #     learning_rate = round(step * 0.0003, 4)
    #     forget_factor = 1 - learning_rate
    #     apa = APA_Non_Stationary(waveData, size, filter_length, order, learning_rate, lamda)
    #     w, e, y_pred = apa.run_apa()
    #     mse_e = e ** 2
    #     sum_mse = np.sum(mse_e) / mse_e.shape[0]
    #     sum_mse_list.append(sum_mse)
    #     forget_factor_list.append(forget_factor)
    #     plt.subplot(4, 4, (step - (15 * (step_list[0] // 15))))
    #     plt.plot(x, mse_e, label='forget_factor={}'.format(forget_factor))
    #     plt.legend(loc='upper right')
    # plt.show()
    #
    # step_list = np.arange(16, 31)
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # for step in step_list:
    #     learning_rate = round(step * 0.0003, 4)
    #     forget_factor = 1 - learning_rate
    #     apa = APA_Non_Stationary(waveData, size, filter_length, order, learning_rate, lamda)
    #     w, e, y_pred = apa.run_apa()
    #     mse_e = e ** 2
    #     sum_mse = np.sum(mse_e) / mse_e.shape[0]
    #     sum_mse_list.append(sum_mse)
    #     forget_factor_list.append(forget_factor)
    #
    #     plt.subplot(4, 4, (step - (15 * (step_list[0] // 15))))
    #     plt.plot(x, mse_e, label='forget_factor={}'.format(forget_factor))
    #     plt.legend(loc='upper right')
    # plt.show()
    #
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # plt.plot(forget_factor_list, sum_mse_list)
    # plt.suptitle("The relationship between MSE and forget factor (filter order={})".format(order), fontsize=18)
    # plt.show()

    # size = waveData.shape[0]
    # lamda = 0.9
    # order = 6
    # learning_rate = 0.0008
    # filter_length = 20
    #
    # rls = RLS(waveData, size, order, forget_factor=(1 - learning_rate))
    # w1, e1, y_pred1 = rls.run_rls()
    #
    # apa = APA_Non_Stationary(waveData, size, filter_length, order, learning_rate, lamda)
    # w2, e2, y_pred2 = apa.run_apa()
    #
    # mse1_list = []
    # mse2_list = []
    # for i in range(w1_list.shape[0]):
    #     print(i)
    #     mse1 = wr.mse(w1[i, :])
    #     mse2 = wr.mse(w2.T[i, :])
    #     mse1_list.append(mse1)
    #     mse2_list.append(mse2)

    # mse
    # x = np.arange(order, (order + y_pred1.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # ax1 = plt.subplot(2, 1, 1)
    # ax1 = plt.plot(x, mse1_list, c='red', label='RLS')
    # plt.legend(loc='best')
    # ax2 = plt.subplot(2, 1, 2)
    # ax2 = plt.plot(x, mse2_list, c='blue', label='APA')
    # plt.legend(loc='best')
    # plt.show()

    # complexity -- filter length
    # time_list =[]
    # filter_length_list = np.arange(1,30)
    # for filter_length in filter_length_list:
    #     print(filter_length)
    #     start = time.time()
    #     apa = APA_Non_Stationary(waveData, size, filter_length, order, learning_rate, lamda)
    #     w, e, y_pred = apa.run_apa()
    #     end = time.time()
    #     running_time = end - start
    #     time_list.append(running_time)
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # plt.plot(filter_length_list, time_list)
    # plt.xlabel('filter length')
    # plt.ylabel('Time')
    # plt.suptitle("The relationship between complexity and filter length (filter order={})".format(order), fontsize=18)
    # plt.show()

    mat = scipy.io.loadmat('./project1.mat')
    primary = mat['primary']
    reference = mat['reference']
    fs = mat['fs']
