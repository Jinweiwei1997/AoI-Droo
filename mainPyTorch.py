#  #################################################################
#  Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks
#
#  This file contains the main code of DROO. It loads the training samples saved in ./data/data_#.mat, splits the samples into two parts (training and testing data constitutes 80% and 20%), trains the DNN with training and validation samples, and finally tests the DNN with test data.
#
#  Input: ./data/data_#.mat
#    Data samples are generated according to the CD method presented in [2]. There are 30,000 samples saved in each ./data/data_#.mat, where # is the user number. Each data sample includes
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       energy broadcasting parameter   |    output_a           |
#  -----------------------------------------------------------------
#  |     transmit time of wireless device  |    output_tau         |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#
#  References:
#  [1] 1. Liang Huang, Suzhi Bi, and Ying-Jun Angela Zhang, "Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks," in IEEE Transactions on Mobile Computing, early access, 2019, DOI:10.1109/TMC.2019.2928811.
#  [2] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177-4190, Jun. 2018.
#
# version 1.0 -- July 2018. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import mat4py

# Implementated based on the PyTorch
import torch

from memoryPyTorch import MemoryDNN
from bisection import bisection

import time

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def plot_rate(rate_his, rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
    #rolling_intv = 20

    plt.plot(np.arange(len(rate_array))+1, df.rolling(rolling_intv, min_periods=1).mean().values, 'b')
    plt.fill_between(np.arange(len(rate_array))+1, df.rolling(rolling_intv, min_periods=1).min()[0], df.rolling(rolling_intv, min_periods=1).max()[0], color = 'b', alpha = 0.2)
    plt.ylabel('Normalized Computation Rate')
    plt.xlabel('Time Frames')
    plt.show()

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''

    N = 5                       # number of users
    n = 120000                    # number of time frames
    K = N                        # initialize K = N
    decoder_mode = 'OP'          # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 1024                # capacity of memory structure
    Delta = 32                   # Update interval for adaptive K

    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))
    # Load data
    data = sio.loadmat('./data/data_%d' %N)
    channel_h = data['input_h']
    channel_g = sio.loadmat('./data/data_%d' %N)['input_g']
    NodeBEnergy = sio.loadmat('./data/data_%d' %N)['input_battery']
    ProcessAoI = sio.loadmat('./data/data_%d' %N)['input_aoi']
    rate = sio.loadmat('./data/data_%d' %N)['output_obj'] # this rate is only used to plot figures; never used to train DROO.
    '''

    channel_h = data["input_h"]
    channel_g = data['input_g']
    BEnergy = data['input_battery']
    AoI = data['input_aoi']
    '''
    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel_h = channel_h * 1000000
    channel_g = channel_g * 1000000
    Energy = NodeBEnergy*1000
    channel = [x for x in channel_h]
    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if n > total data size

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n)) # training data size

    Action =[]
    mem = MemoryDNN(net = [4*N, 120, 80, N+1],
                    learning_rate = 0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory
                    )

    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []
    for i in range(n):
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        if i> 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(k_idx_his[-Delta:-1]) +1;
            else:
                max_k = k_idx_his[-1] +1;
            K = min(max_k +1, N)

        if i < n - num_test:
            # training
            i_idx = i % split_idx
        else:
            # test
            i_idx = i - n + num_test + split_idx

        h = channel_h[i_idx,:]
        g = channel_g[i_idx,:]
        AoI = ProcessAoI[i_idx,:]
        BEnergy = Energy[i_idx,:]

        # the action selection must be either 'OP' or 'KNN'
        m_list = mem.decode(h, N, decoder_mode)

        r_list = []
        for m in m_list:
            r_list.append(bisection(h/1000000,g/1000000,BEnergy/1000,AoI, m)[0])

        # encode the mode with largest reward
        mem.encode(h,g,BEnergy,AoI, m_list[np.argmax(r_list)])
        # the main code for DROO training ends here
        count = 0 #start the calculate AverageAoI



        # the following codes store some interested metrics for illustrations
        # memorize the largest reward
        rate_his.append(np.max(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        # record the index of largest reward
        k_idx_his.append(np.argmax(r_list))
        # record K in case of adaptive K
        K_his.append(K)
        mode_his.append(m_list[np.argmax(r_list)])
    total_time = time.time() - start_time
    mem.plot_cost()
    plot_rate(rate_his_ratio)

    print("Averaged normalized computation rate:", sum(rate_his_ratio[-num_test: -1]) / num_test)
    print('Total time consumed:%s' % total_time)
    print('Average time per channel:%s' % (total_time / n))

    # save data into txt
    save_to_txt(k_idx_his, "k_idx_his.txt")
    save_to_txt(K_his, "K_his.txt")
    save_to_txt(mem.cost_his, "cost_his.txt")
    save_to_txt(rate_his_ratio, "rate_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")

    #####start test
    ac = 0
    AoI_t=[1,1,1,1,1]
    BEnergy_t = NodeBEnergy[1, :]

    Amax = 6
    Bmax = 0.0004
    sigma = 3.162277660168375 * 10 ** (-13)
    S = 12
    theta = []  # never used 权重
    eta = 0.5  # gain loss
    P = 5.012
    FinalAoI=0
    for i in range(100):
        AverSumAoI = 0
        h_t = channel_h[i, :]
        g_t = channel_g[i, :]
        predict = mem.model(torch.Tensor(np.hstack((h_t,g_t,BEnergy_t*1000,AoI_t))))
        ac += 1
        flat = 0
        EnergyHarvest = [0 for j in range(N)]  # amount of energy harvest
        maxj=torch.Tensor.argmax(predict)
        for j in range(N+1):
            if maxj==0:
                for j in range(N):
                    EnergyHarvest[j] = eta * P * g_t[j]/1000000
                for k in range(N):
                    if AoI_t[k] < Amax:
                        AoI_t[k] += 1
                for j in range(N):
                    if EnergyHarvest[j] + BEnergy_t[j] < Bmax:
                        BEnergy_t[j] += EnergyHarvest[j]
                    else:
                        BEnergy_t[j] = Bmax
            else:
                if j == maxj:
                    flat = 1
                    AoI_t[j-1] = 1
                    EnergyTrans = sigma / h_t[j-1] * (2 ** S)
                    if EnergyTrans > BEnergy_t[j-1]:
                        print("传输能量比现有的能量高")
                    BEnergy_t -= EnergyTrans
                    for k in range(N):
                        if k != j and AoI_t[k] < Amax:
                            AoI_t[k] += 1
                    break

        for j in range(N):
            AverSumAoI +=AoI_t[j]
        AverSumAoI /= N
        FinalAoI = (FinalAoI *i +AverSumAoI)/(i+1)
    print("Aoi:",FinalAoI)
