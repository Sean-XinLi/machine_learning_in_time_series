import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, signal

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


def klms(input, desired, order_number, L, sigma, sigma_prime, lr):
    length = input.shape[0]
    X_time = np.zeros([length, order_number])
    for i in range(length):
        index = i
        for j in range(order_number):
            X_time[i][j] = input[index]
            index -= 1
            if (index < 0):
                break
    output = np.zeros([1, length])

    MSE_curve = np.zeros(length)

    inputs = np.zeros([L + length, order_number])
    errors = np.zeros(L + length)
    weights = np.zeros(L + length)

    w_list = np.zeros((L + length,length))

    for i in range(length):
        print(i)
        phi = np.exp(-(((inputs[0:L + i] - X_time[i]) ** 2).sum(1)) / (2 * sigma * sigma))
        cur_output = np.dot(weights[0:L + i], phi)
        cur_err = desired[i] - cur_output
        errors[L + i] = cur_err
        inputs[L + i] = X_time[i]
        cur_weight = (
                (np.exp(-(((errors[i + 1:L + i + 1] - errors[L + i]) ** 2)) / (2 * sigma_prime * sigma_prime))) * (
                errors[L + i] - errors[i + 1:L + i + 1])).sum()
        weights[L + i] = cur_weight * lr / (L * sigma_prime * sigma_prime)
        phi = np.exp(-(((inputs[0:L + i + 1] - X_time[i]) ** 2).sum(1)) / (2 * sigma * sigma))
        w_list[L + i,i] = weights[L + i]

        output[0][i] = np.dot(weights[0:L + i + 1], phi)
        MSE_curve[i] = errors[0:L + i + 1].sum() / (i + 1)

    return output, errors, w_list

def prediction2(input,errors, desired,order_number, L, sigma, sigma_prime, lr):
    length = input.shape[0]
    X_time = np.zeros([length, order_number])
    for i in range(length):
        index = i
        for j in range(order_number):
            X_time[i][j] = input[index]
            index -= 1
            if (index < 0):
                break
    output = np.zeros([1, length])

    MSE_curve = np.zeros(length)

    inputs = np.zeros([L + length, order_number])
    errors = np.zeros(L + length)
    weights = np.zeros(L + length)

    w_list = np.zeros((L + length, length))

    for i in range(1000):
        print(i)
        phi = np.exp(-(((inputs[0:L + i] - X_time[i]) ** 2).sum(1)) / (2 * sigma * sigma))
        cur_output = np.dot(weights[0:L + i], phi)
        cur_err = desired[i] - cur_output
        # errors[L + i] = cur_err
        inputs[L + i] = X_time[i]
        cur_weight = (
                (np.exp(-(((errors[i + 1:L + i + 1] - errors[L + i]) ** 2)) / (2 * sigma_prime * sigma_prime))) * (
                errors[L + i] - errors[i + 1:L + i + 1])).sum()
        weights[L + i] = cur_weight * lr / (L * sigma_prime * sigma_prime)
        phi = np.exp(-(((inputs[0:L + i + 1] - X_time[i]) ** 2).sum(1)) / (2 * sigma * sigma))
        weights[L + i] = w_list[L + i, i]
        output[0][i] = np.dot(weights[0:L + i + 1], phi)

    return output

t = np.arange(0, 2, 0.0001)
input = np.sin(2000 * np.pi * t)

desired = np.sin(4000 * np.pi * t)
mix = np.random.binomial(1, 0.9, t.shape[0])
noise = np.zeros(t.shape[0])
for i in range(t.shape[0]):
    noise[i] = mix[i] * np.random.normal(0, 0.1) + (1 - mix[i]) * np.random.normal(4, 0.1)

desired = noise + desired

order_number = 10
L = 500
sigma = 0.1
sigma_prime = [0.01, 0.1, 1, 10]
lr = [0.1, 0.4, 0.7, 0.9]

y8, e8, w_list = klms(input, desired, order_number, L, sigma, sigma_prime[2], lr[3])

freq4 = [500, 1000, 1500, 2000]
fig = plt.figure(figsize=(16, 9), dpi=100)

for step in range(4):
    input_data = np.sin(2 * np.pi * freq4[step] * t)
    y6 = prediction2(input,e8, desired,order_number,L,sigma,sigma_prime[2],lr[3])
    plt.subplot(4, 1, (step + 1))
    plt.plot(y6[0], label='y_pred,input freq = {}'.format(freq4[step]))
    plt.plot(desired, label='desired')
    plt.legend(loc="upper right")
    plt.xlim(0, 200)

plt.suptitle("performance of prediction signal in different input data frequency in KLMS-MEE ")
plt.savefig("1.png")
plt.show()








