import numpy as np
import matplotlib.pyplot as plt

filtpath = "./sun_spot_input.asc"
input_data = np.loadtxt(filtpath,skiprows=1)
print(input_data.shape)

power = (input_data ** 2).sum() / input_data.shape[0]
# plt.plot(input_data)
# plt.show()

# KRLS-MSE

order = 6
delay = 1
lamda = 0.1
forget_factor = 1
kernel_size = 0.0001

desired = input_data[order+delay:]

length = input_data.shape[0] - order - delay
e = np.zeros(length)
one = np.ones(1).reshape(1,1)
Q_temp = 1/((np.array((lamda * forget_factor + 1)).reshape(1,1)))
a_temp = Q_temp * desired[0]

save_matrix = np.zeros((length,order))
for i in range(length):
    save_matrix[i] = input_data[i:i+order]

for step in range(1, length):
    print(step)
    input_seg = save_matrix[step, :]
    G = np.exp(-kernel_size * ((save_matrix[:step] - input_seg) ** 2).sum(1)).reshape(-1, 1)

    Z = Q_temp @ G

    r = (lamda * (forget_factor ** step) + 1 - Z.T @ G)
    Q_temp = Q_temp * r + Z * Z.T

    Q_temp = np.hstack((Q_temp, -Z))

    temp = np.hstack((-Z.T, one))

    Q_temp = np.vstack((Q_temp, temp))

    Q_temp = Q_temp / r
    y_temp = np.dot(G.T, a_temp)
    e[step] = desired[step] - y_temp

    temp2 = a_temp - Z / r * e[step]
    a_temp = np.vstack((temp2, e[step] / r))
