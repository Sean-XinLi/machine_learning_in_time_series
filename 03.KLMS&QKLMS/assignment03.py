import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import time


class GenData(object):
    def __init__(self, size):
        # how many samples in data
        self.size = size

        # fix random value
        np.random.seed(1)
        # generate (u=0,s^2=1) gaussian random variable
        self.noisy1 = np.random.normal(0, 1, self.size)

        self.desired_data = self.noisy1

    def filtering_data(self):
        # step 1 ==> through linear filter (tn = -0.8 * yn + 0.7 * yn * z^-1)
        t_in = np.arange(0, self.size)

        # [-0.8, 0.7] is numerator, [1, 0] is denominator, t is dt or step
        system = ([-0.8, 0.7], [1, 0], 1)

        # signal.dlsim is used for Simulating output of a discrete-time linear system
        # self.noisy is input data here, t mean time step, t is optional
        # self.t is time value for the output, self.tn is system response
        self.t, self.tn = signal.dlsim(system, self.noisy1, t=t_in)
        self.tn = self.tn.reshape(self.size)

        # step 2 ===>through nonlinear filter (qn = tn + 0.25 tn ^2 + 0.11 tn^3)
        self.qn = self.tn + 0.25 * self.tn ** 2 + 0.11 * self.tn ** 3

        return self.qn

    def get_data(self):
        # signal qn is corrupted by 15dB AWGN -- output = qn + noise
        # SNR = 10 log_10(P / N)  -- P is the power of input, N is the power of noise
        # step 1, compute the power of input
        self.qn = self.filtering_data()
        self.P = np.sum(self.qn ** 2) / self.size

        # step 2， compute the power of noise based on 15dB(SNR)
        self.N = self.P / 10 ** 1.5

        # fix random value
        np.random.seed(1)
        self.noisy2 = np.random.normal(0, self.N, self.size)

        # step 3, output = qn + noise
        self.input_data = self.qn + self.noisy2
        # self.input_data = self.qn
        self.data = np.vstack((self.input_data, self.desired_data))

        return self.data


class LMS_family():
    def __init__(self, data, learning_rate, time_delay, filter_length, threshold, c, mode=1):
        self.input = data[0, :]
        self.desire = data[1, :]
        self.order = filter_length
        self.time_delay = time_delay
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.mode = mode

        # data preprocessing
        self.size = data.shape[1]
        self.power = np.sum(self.input ** 2) / self.size  # power of input

        # compute inter-quartile
        top_quantile_input = np.quantile(self.input, 0.75)
        bottom_quantile_input = np.quantile(self.input, 0.25)
        inter_quartile = abs(top_quantile_input - bottom_quantile_input)

        # rule of thumb for h (h = 1.06 min(input data s.d. or inter-quartile /1.34) * N ** -1/(5L))
        # self.h = c * np.min((np.sqrt(self.power), inter_quartile)) * self.size ** (-0.2 / self.order)
        # print(self.h)
        self.h = c

    def run_LMS(self):
        count_list = np.zeros(self.size)
        count = self.order
        # predict output will start at the data size index = order
        length = self.size - self.order

        # initial arguments - y ==> predict output, e ==> error, w ==> weights
        y = np.zeros(length)
        e = np.zeros(length)
        w = np.zeros((length, self.order))

        for i in range(length):
            count = count + 1
            count_list[i + self.order] = count

            # initial w0
            if i == 0:
                w_temp = np.zeros(self.order)
            # predict output = w0 * x(n) + w1 * x(n-1) + ...  ==> reverse the segment of input data
            input_seg = self.input[i:(i + self.order)]
            input_seg = input_seg[::-1]
            y_pred = np.dot(w_temp, input_seg)

            # compute error
            e_temp = self.desire[i + self.order - self.time_delay] - y_pred

            # update weights
            w_temp = w_temp + self.learning_rate * e_temp * input_seg

            # collect result
            y[i] = y_pred
            e[i] = e_temp
            w[i, :] = w_temp.T

        # return e, y, count_list

        return y, e, count_list

    def run_KLMS(self):
        count_list = np.zeros(self.size)
        count = self.order

        # predict output will start at the data size index = order
        length = self.size - self.order
        # initial arguments - y ==> predict output, e ==> error
        e = np.zeros(length)
        y = np.zeros(length)
        w = np.zeros(self.size)

        for step in range(length):

            count = count + 1
            count_list[step] = count

            # put input into RKHS, 1/h  is kernel size
            # input data will be translate to H space, H_space_input will be new input data in KLMS
            # φ1 is 5-dimension vector
            # e(i) = d(i) - sum(learning_rate * e(0:step) * k<φ(0:step),φ(step)>)
            # k<φ1,φ2> is inner produce of φ1 and φ2, k<φ1,φ2> = G(φ1-φ2) is scalar

            # consider time series, prediction only related to filter order
            # e.g. filter_order = 2 , prediction only related to x(n),x(n-1)

            # input_seg2 and input_seg1 can be instead by (sample[step]-sample[0:step]) to avoid repeated computation
            self.input_seg2 = self.input[step:(step + self.order)]
            y_temp = 0  # for prediction output
            for i in range(step):
                self.input_seg1 = self.input[i:(i + self.order)]

                # dist = |x(i)-x(step)|
                dist1 = np.linalg.norm((self.input_seg1 - self.input_seg2), ord=self.mode)
                G = np.exp(-self.h * (dist1) ** 2)

                # output y(n) should match d(n - 2), start index is order
                y_temp = y_temp + self.learning_rate * e[i] * G

            # y_temp = y_temp * 100
            e_temp = self.desire[step + self.order - self.time_delay] - y_temp

            # # if kernel is gaussian, k<φ(step,·)means input data at index step input_data[step] is Gaussian's mean
            # # using np matrix broadcast tech. sum them in columns.

            # # (optional) for get trace of w, next two line can be comment for optimal
            # dist2 = np.linalg.norm((self.input.reshape(-1, 1) - self.input_seg2.reshape(1, -1)), ord=self.mode, axis=1)
            # H_space_up_limit = np.exp(-self.h * (dist2) ** 2)

            # for loop would not work when step == 0
            if step == 0:
                # (optional) for get trace of w, next one line can be comment for optimal
                # e_temp = self.desire[step+self.order-self.time_delay] - np.dot(w, H_space_up_limit)
                e_temp = self.desire[step + self.order - self.time_delay]

            e[step] = e_temp
            y[step] = y_temp

            # (optional) update weights,for get trace of w, next one line can be comment for optimal
            # w = w + self.learning_rate * e_temp * H_space_up_limit

        # return e, y, count_list
        return y, e, count_list

    def run_QKLMS(self):
        # define count
        count = self.order
        count_list = np.zeros([count])

        # initial center dictionary, element in dictionary is vector, the number of vector based on filter order.
        # initial a list of coefficient vector alpha = learning_rate * e(i)
        # according to the weight update equation w = w + self.learning_rate * e[i] * φ(i,·)
        # a = self.learning_rate * e[i], a is the element in alpha
        # center dictionary and a list of coefficient vector alpha should have the same shape
        # when i == 0, e(0) == d(0). because e(i) = d(i) - <w(i-1),φ(i)>_F , all the element in w(0) is zero.

        # consider time series, prediction only related to filter order
        # e.g. filter_order = 2 , prediction only related to x(n),x(n-1)
        element = self.input[0:self.order]
        center_list = np.array([element])
        alpha = np.array([self.learning_rate * self.desire[3]])

        result = np.zeros(1)

        # predict output will start at the data size index = order
        length = self.size - self.order

        for step in range(length):
            element = self.input[step:(step + self.order)]

            # # if kernel is gaussian, k<φ(step,·)means input data at index step input_data[step] is Gaussian's mean
            # # using np matrix broadcast tech. sum them in columns.

            # # (optional) for get trace of w, next two line can be comment for optimal
            # dist2 = np.linalg.norm((self.input.reshape(-1, 1) - element(1, -1)), ord=self.mode, axis=1)
            # H_space_up_limit = np.exp(-self.h * (dist2) ** 2)

            # for-loop would not work when step == 0, we can initialize
            if step == 0:
                # initial arguments - y ==> predict output, e ==> error, w ==> weights
                e = np.zeros(length)
                y = np.zeros(length)
                w = np.zeros(self.size)

                # # (optional) for get trace of w, next one line can be comment for optimal
                # e_temp = self.desire[step+self.order-self.time_delay] - np.dot(w, H_space_up_limit)
                e_temp = self.desire[step + self.order - self.time_delay]
                e[0] = e_temp
                result[0] = 0

                # # (optional) update weights,for get trace of w, next one line can be comment for optimal
                # w = w + self.learning_rate * e_temp * H_space_up_limit

            # compare the distance between the new coming data and each element in center_list, find the minimal one
            dist_list = np.linalg.norm((center_list - element), ord=self.mode, axis=1)
            dist_min = np.min(dist_list)

            # get the minimal dist index in dist_list or center_list
            # this index is the point which is very close to the new coming data
            min_index = np.argmin(dist_list)

            # computing coefficient vector, w = w + self.learning_rate * e[i] * φ(i,·)
            a = self.learning_rate * e[step - 1]

            if dist_min <= self.threshold:
                # if dist_min is too small, which mean this point is very close to one point of the center_list
                # for time complexity, do not count this point，but still need the information of this point
                alpha[min_index] = alpha[min_index] + a

                y_temp = result[min_index] + a * np.exp(-self.h * (dist_list[-1]) ** 2)
                e_temp = self.desire[step + self.order - self.time_delay] - y_temp

            else:
                # if dist_min is npt too small, which mean this point should be count
                count = count + 1

                y_temp = 0  # for prediction output

                # put input into RKHS, 1/h  is kernel size
                # input data will be translate to H space, H_space_input will be new input data in KLMS
                # e(i) = d(i) - sum(learning_rate * e(0:step) * k<φ(0:step),φ(step)>)
                # k<φ1,φ2> is inner produce of φ1 and φ2, k<φ1,φ2> = G(φ1-φ2) is scalar

                # ps: center_list.shape[0]-1, because the lase element just add into center_list, should not be consider
                for index in range(center_list.shape[0]):
                    G = np.exp(-self.h * (dist_list[index]) ** 2)
                    # output y(n) should match d(n - 2), start index is order
                    y_temp = y_temp + alpha[index] * G

                result = np.hstack((result, y_temp))

                e_temp = self.desire[step + self.order - self.time_delay] - y_temp

                # y[step] = y_temp

                # add this point to zhe center_list and add a to the alpha
                center_list = np.vstack((center_list, element))
                alpha = np.hstack((alpha, a))

                # # (optional) update weights,for get trace of w, next one line can be comment for optimal
                # w = w + alpha[-1] * H_space_up_limit
            y[step] = y_temp
            e[step] = e_temp
            count_list = np.hstack((count_list, count))
        print(count)
        # return e, y, count_list
        return y, e, count_list


if __name__ == '__main__':
    # set parameter
    data_size = 5000
    time_delay = 2  # output y(n) should match d(n-2)
    filter_length = 5  # order
    learning_rate1 = 0.01  # for lms
    learning_rate2 = 0.0001  # learning_rate for KLMS and QKLMS, should be less than 1
    c = 0.1  # kernel size
    threshold = 1.5

    # collect data,the first row is input_data xn ,the second row is desired_data yn
    gen = GenData(data_size)
    data = gen.get_data()

    exp1 = LMS_family(data, learning_rate1, time_delay, filter_length, threshold, c)
    y1, e1, count1_list = exp1.run_LMS()
    exp2 = LMS_family(data, learning_rate2, time_delay, filter_length, threshold, c)
    y2, e2, count2_list = exp2.run_KLMS()
    y3, e3, count3_list = exp2.run_QKLMS()
    x = np.arange(filter_length, data_size)
    fig = plt.figure(figsize=(16, 9), dpi=100)

    plt.subplot(4, 1, 1)
    ax1 = plt.plot(x, data[1, :][filter_length:data_size], c='red')
    plt.xlabel("iteration(desired)")
    plt.ylabel("signal")
    plt.grid(1)
    fig.tight_layout()

    plt.subplot(4, 1, 2)
    ax2 = plt.plot(x, y1, c='green')
    plt.ylim(-4, 4)
    plt.xlabel("iteration(LMS)")
    plt.ylabel("prediction signal")
    plt.grid(1)
    fig.tight_layout()

    plt.subplot(4, 1, 3)
    ax3 = plt.plot(x, y2, c='blue')
    plt.xlabel("iteration(KLMS)")
    plt.ylabel("prediction signal")
    plt.grid(1)
    fig.tight_layout()

    plt.subplot(4, 1, 4)
    ax4 = plt.plot(x, y3, c='pink')
    plt.xlabel("iteration(QLMS)")
    plt.ylabel("prediction signal")
    plt.grid(1)
    plt.suptitle("performance of prediction in different LMS family"
                 "(learning rate={} for LMS,learning rate={} for KLMS&QLMS ,"
                 "kernel size={},network size={})".format(learning_rate1, learning_rate2, c, int(count3_list[-1])))
    fig.tight_layout()
    # plt.savefig("matching performance.png")
    plt.show()

    # kernel size test
    # c1 = 0.001
    # exp = LMS_family(data, learning_rate2, time_delay, filter_length, threshold, c1)
    # y1, e1, count1_list = exp.run_KLMS()
    # c2 = 0.01
    # exp = LMS_family(data, learning_rate2, time_delay, filter_length, threshold, c2)
    # y2, e2, count2_list = exp.run_KLMS()
    #
    # c3 = 0.1
    # exp = LMS_family(data, learning_rate2, time_delay, filter_length, threshold, c3)
    # y3, e3, count3_list = exp.run_KLMS()
    #
    # mse1_list = np.zeros(data_size - filter_length)
    # mse2_list = np.zeros(data_size - filter_length)
    # mse3_list = np.zeros(data_size - filter_length)
    #
    # e1_sqrt = e1 ** 2
    # e2_sqrt = e2 ** 2
    # e3_sqrt = e3 ** 2
    #
    # for step in range(data_size - filter_length):
    #     if step == 0:
    #         mse1_list[step] = e1_sqrt[step]
    #         mse2_list[step] = e2_sqrt[step]
    #         mse3_list[step] = e3_sqrt[step]
    #
    #     mse1_list[step] = (mse1_list[step - 1] * step + e1_sqrt[step]) / (step + 1)
    #     mse2_list[step] = (mse2_list[step - 1] * step + e2_sqrt[step]) / (step + 1)
    #     mse3_list[step] = (mse3_list[step - 1] * step + e3_sqrt[step]) / (step + 1)
    #
    # x = np.arange(filter_length, (filter_length + mse1_list.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # ax1 = plt.plot(x, mse1_list, c='red', label='KLMS1={}'.format(c1))
    # ax2 = plt.plot(x, mse2_list, c='green', label='KLMS1={}'.format(c2))
    # ax3 = plt.plot(x, mse3_list, c='blue', label='KLMS1={}'.format(c3))
    # plt.legend(loc="upper right")
    # plt.xlabel("iteration")
    # plt.ylabel("mse")
    # plt.title("performance of KLMS in different kernel size")
    # plt.grid(1)
    # plt.savefig("kernal_size.png")
    # plt.show()

    # （2）
    # exp1 = LMS_family(data, learning_rate1, time_delay, filter_length, threshold, c)
    # y1, e1, count1_list = exp1.run_LMS()
    # exp2 = LMS_family(data, learning_rate2, time_delay, filter_length, threshold, c)
    # y2, e2, count2_list = exp2.run_KLMS()
    # y3, e3, count3_list = exp2.run_QKLMS()
    #
    # # #
    # mse1_list = np.zeros(data_size - filter_length)
    # mse2_list = np.zeros(data_size - filter_length)
    # mse3_list = np.zeros(data_size - filter_length)
    # #
    # e1_sqrt = e1 ** 2
    # e2_sqrt = e2 ** 2
    # e3_sqrt = e3 ** 2
    # #
    # for step in range(data_size - filter_length):
    #     if step == 0:
    #         mse1_list[step] = e1_sqrt[step]
    #         mse2_list[step] = e2_sqrt[step]
    #         mse3_list[step] = e3_sqrt[step]
    # #
    #     mse1_list[step] = (mse1_list[step - 1] * step + e1_sqrt[step]) / (step + 1)
    #     mse2_list[step] = (mse2_list[step - 1] * step + e2_sqrt[step]) / (step + 1)
    #     mse3_list[step] = (mse3_list[step - 1] * step + e3_sqrt[step]) / (step + 1)
    # # #
    # x = np.arange(filter_length, (filter_length + mse1_list.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # ax1 = plt.plot(x, mse1_list, c='red', label='LMS')
    # ax2 = plt.plot(x, mse2_list, c='green', label='KLMS')
    # ax3 = plt.plot(x, mse3_list, c='blue', label='QKLMS')
    # # plt.ylim(0, 1)
    # plt.legend(loc="upper right")
    # plt.xlabel("iteration")
    # plt.ylabel("mse")
    # plt.title("the MSE in LMS family (learning rate={} for LMS,learning rate={} "
    #           "for KLMS&QLMS ,kernel size={},network size={})"
    #           .format(learning_rate1, learning_rate2, c, int(count3_list[-1])))
    # plt.show()

    # growth curve

    # exp1 = LMS_family(data, learning_rate1, time_delay, filter_length, threshold, c)
    # y1, e1, count1_list = exp1.run_LMS()
    # exp2 = LMS_family(data, learning_rate2, time_delay, filter_length, threshold, c)
    # y2, e2, count2_list = exp2.run_KLMS()
    # y3, e3, count3_list = exp2.run_QKLMS()
    #
    # mse1_list = np.zeros(data_size - time_delay)
    # mse2_list = np.zeros(data_size - time_delay)
    # mse3_list = np.zeros(data_size - time_delay)
    #
    # e1_sqrt = e1 ** 2
    # e2_sqrt = e2 ** 2
    # e3_sqrt = e3 ** 2
    # for step in range(data_size - filter_length):
    #     if step == 0:
    #         mse1_list[step] = e1_sqrt[step]
    #         mse2_list[step] = e2_sqrt[step]
    #         mse3_list[step] = e3_sqrt[step]
    #     mse1_list[step] = (mse1_list[step - 1] * step + e1_sqrt[step]) / (step + 1)
    #     mse2_list[step] = (mse2_list[step - 1] * step + e2_sqrt[step]) / (step + 1)
    #     mse3_list[step] = (mse3_list[step - 1] * step + e3_sqrt[step]) / (step + 1)
    #
    # x = np.arange(filter_length, (filter_length + mse3_list.shape[0]))
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # plt.subplot(2, 1, 1)
    # ax1 = plt.plot(x, mse1_list, c='red', label='LMS')
    # ax2 = plt.plot(x, mse2_list, c='green', label='KLMS')
    # ax3 = plt.plot(x, mse3_list, c='blue', label='QKLMS')
    # plt.legend(loc="upper right")
    # plt.xlabel("iteration")
    # plt.ylabel("mse")
    # plt.title("performance of LMS, KLMS, QKLMS in kernel size = {},"
    #           "step size for LMS = {},step size for KLMS & QKLMS = {}".format(c, learning_rate1, learning_rate2))
    # fig.tight_layout()
    # plt.subplot(2, 1, 2)
    # x = np.arange(1, 5001)
    # ax1 = plt.plot(x, count1_list, c='red', label='LMS')
    # ax2 = plt.plot(x, count2_list, c='green', label='KLMS')
    # ax3 = plt.plot(x, count3_list, c='blue', label='QKLMS')
    # plt.legend(loc="upper right")
    # plt.xlabel("iteration")
    # plt.ylabel("network size")
    # plt.title("growth curve")
    # fig.tight_layout()
    # plt.savefig("growth_curve.png")
    # plt.show()

    # performance of QKLMS in different network size
    # exp2 = LMS_family(data, learning_rate2, time_delay, filter_length, threshold, c)
    # y2, e2, count2_list = exp2.run_KLMS()
    # mse2_list = np.zeros(data_size - filter_length)
    # e2_sqrt = e2 ** 2
    # for step in range(data_size - filter_length):
    #     if step == 0:
    #         mse2_list[step] = e2_sqrt[step]
    #
    #     mse2_list[step] = (mse2_list[step - 1] * step + e2_sqrt[step]) / (step + 1)
    # np.save("klms_mse_data", mse2_list)
    #
    # # find the best number of network size
    # mse2_list = np.load("klms_mse_data.npy")
    # x = np.arange(filter_length, data_size)
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # for i in range(8):
    #     threshold = 0.5 * (i+1)
    #     exp2 = LMS_family(data, learning_rate2, time_delay, filter_length, threshold, c)
    #     y3, e3, count3_list = exp2.run_QKLMS()
    #     mse3_list = np.zeros(data_size - filter_length)
    #     e3_sqrt = e3 ** 2
    #
    #     for step in range(data_size - filter_length):
    #         if step == 0:
    #             mse3_list[step] = e3_sqrt[step]
    #         mse3_list[step] = (mse3_list[step - 1] * step + e3_sqrt[step]) / (step + 1)
    #     plt.subplot(2, 4, (i + 1))
    #     ax1 = plt.plot(x, mse2_list, c='red', label='KLMS')
    #     ax2 = plt.plot(x, mse3_list, c='green', label='QKLMS')
    #     plt.legend(loc="upper right")
    #     plt.xlabel("iteration")
    #     plt.ylabel("MSE")
    #     plt.title("network size = {}".format(count3_list[-1]))
    #     fig.tight_layout()
    # plt.suptitle("comparing the mse between KLMS and QKLMS in different network size")
    # fig.tight_layout()
    # plt.savefig("2.png")
    # plt.show()

    # running time
    # time3_list = []
    # count4_list = []
    #
    # exp1 = LMS_family(data, learning_rate1, time_delay, filter_length, threshold, c)
    # tic1 = time.perf_counter()
    # y1, e1, count1_list = exp1.run_LMS()
    # toc1 = time.perf_counter()
    #
    # exp2 = LMS_family(data, learning_rate2, time_delay, filter_length, threshold, c)
    # time1 = toc1 - tic1
    # tic2 = time.perf_counter()
    # y2, e2, count2_list = exp2.run_KLMS()
    # toc2 = time.perf_counter()
    # time2 = toc2 - tic2
    #
    # for step in range(8):
    #     threshold = 0.5 * step
    #     exp2 = LMS_family(data, learning_rate2, time_delay, filter_length, threshold, c)
    #     tic3 = time.perf_counter()
    #     y3, e3, count3_list = exp2.run_QKLMS()
    #     toc3 = time.perf_counter()
    #     time3 = toc3 - tic3
    #     count4_list.append(count3_list[-1])
    #     time3_list.append(time3)
    #     print(count3_list)
    # print(count4_list)
    # print(time3_list)
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # ax3 = plt.plot(count4_list, time3_list, c='blue', label='QKLMS')
    # plt.xlabel("network size")
    # plt.ylabel("running time(sound)")
    # plt.title("relationship of running time and network size，running time of LMS = {}s,running time of KLMS = {}s,"
    #           .format(round(time1, 2), round(time2, 2)))
    # plt.legend(loc="upper right")
    # plt.savefig("3.png")
    # plt.show()
