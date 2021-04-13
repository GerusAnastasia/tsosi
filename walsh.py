import numpy as np
import matplotlib.pyplot as plt

def fwt_frequency(a, direction=1):
    if (len(a) == 1):
        return a
    n = len(a)
    b = []
    c = []
    for j in range(n // 2):
        b.append(a[j] + a[j + n // 2])
        c.append(a[j] - a[j + n // 2])
    y = []
    for (i,j) in zip(fwt_frequency(b, direction), fwt_frequency(c, direction)):
        y.append(i)
        y.append(j)
    return y

def conf_fwt_frequency(input_data, direction):
    if direction == -1:
        return [x / len(input_data) for x in fwt_frequency(input_data, direction)]
    else:
        return fwt_frequency(input_data, direction)  

def dwht(input_data, direction=1):
    input_length = len(input_data)

    num_bits = int(np.log2(input_length)) 

    hadamard_h1 = np.array([[1, 1], [1, -1]])
    res_hadamard_matrix = []
    for i in range(num_bits - 1):
        if i == 0:
            res_hadamard_matrix = np.kron(hadamard_h1, hadamard_h1)
        else:
            res_hadamard_matrix = np.kron(res_hadamard_matrix, hadamard_h1)

    result = np.dot(res_hadamard_matrix, input_data)

    if direction == 1:
        result /= input_length

    return result      

def dwt(data, direction=1):
    time_offset = 0.005
    length = len(data)

    transformed_result = []
    temp = 0
    for n in range(length):
        temp = 0
        for i in range(length):
            if direction == 1:
                temp += (data[i] * walsh(n, i / length + time_offset, length)) / length
            else:
                temp += data[i] * walsh(i, n / length + time_offset, length) 
        transformed_result.append(temp)

    return transformed_result

def walsh(n, t, length):
    r = int(np.log2(length))
    rademacher_values= []
    for k in range(1, r + 1): 
        values = rademacher(t, k) ** np.logical_xor(bit_num(n, k - 1), bit_num(n, k))
        rademacher_values.append(values)
    result = 1
    for i in range(0,len(rademacher_values)):
        result *= rademacher_values[i]
    return result

def rademacher(t, k):
    r = np.sin(2 ** k * np.pi * t)
    if r > 0:
        return 1
    else:
        return -1   

def bit_num(value, position):
    mask = 1
    mask <<= position
    if (value == 0 or value & mask == 0):
        return 0
    else:
        return 1 

def main():
    plt.rcParams["figure.figsize"] = (15, 10)
    n = 16
    arguments = np.arange(0, n) * np.pi / 6
    function_values = list(map(lambda x: np.sin(3 * x) + np.cos(x), arguments))

    dwt_res = dwt(function_values, 1)
    fwht_res = conf_fwt_frequency(function_values, 1)
    reverse_dwt_res = dwt(dwt_res, -1)
    reverse_fwht_res = conf_fwt_frequency(fwht_res, -1)

    # plotting part
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

    ax1.plot(arguments, function_values)
    ax1.set(title='Function plot')
    ax1.grid()

    ax2.plot(arguments, function_values)
    ax2.set(title='Function plot')
    ax2.grid()

    ax3.plot(arguments, dwt_res)
    ax3.set(title='Discrete walsh transform (DWT)')
    ax3.grid()

    ax4.plot(arguments, fwht_res)
    ax4.set(title='Fast discrete walsh transform (FWHT)')
    ax4.grid()

    ax5.plot(arguments, reverse_dwt_res)
    ax5.set(title='Reverse discrete walsh transform (RDWT)')
    ax5.grid()

    ax6.plot(arguments, reverse_fwht_res)
    ax6.set(title='Reverse fast discrete walsh transform (RFWHT)')
    ax6.grid()

    plt.show()

if __name__ == '__main__':
    main()
