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
def plot_AoI(rate_his, rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    mpl.style.use('seaborn')
    plt.plot(np.arange(len(rate_array)) + 1, rate_his)
    plt.ylabel('Averagy Sum AoI')
    plt.xlabel('Time Frames')
    plt.show()
'''def plot_rate(rate_his, rolling_intv=50):
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
'''
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

    N = 3                        # number of users
    n = 40000                    # number of time frames
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
    channel_h = channel_h * 10000
    channel_g = channel_g * 10000
    Energy = NodeBEnergy*1000
    channel = [x for x in channel_h]
    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if n > total data size

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n)) # training data size

    Action =[]
    mem = MemoryDNN(net = [4*N,120, 80, N+1],
                    learning_rate = 0.000008,
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
    Energy_train = [0.2,0.2,0.2]
    AoI_text=[]
    AoI = [1, 1, 1]
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
        #AoI = ProcessAoI[i_idx,:]

        BEnergy = Energy[i_idx,:]
        # the action selection must be either 'OP' or 'KNN'
        m_list=[]
        m_list = mem.decode(h,g,Energy_train,AoI, N, decoder_mode)
        r_list = []
        Energy_now = [x / 1000 for x in Energy_train]
        for m in m_list:
            r_list.append(bisection(h / 10000, g / 10000, Energy_now, AoI, m)[0])
        # encode the mode with largest reward
        try:
            Energy_bb = [x for x in (bisection(h / 10000, g / 10000, Energy_now, AoI, m_list[np.argmax(r_list)])[2])]
            Energy_train = [x * 1000 for x in Energy_bb]
            AoI = [x for x in (bisection(h / 10000, g / 10000, Energy_now, AoI, m_list[np.argmax(r_list)])[3])]
            mem.encode(h, g, Energy_train, AoI, m_list[np.argmax(r_list)])

            # the main code for DROO training ends here
            count = 0  # start the calculate AverageAoI

            # the following codes store some interested metrics for illustrations
            # memorize the largest reward
            rate_his.append(np.max(r_list))
            rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
            # record the index of largest reward
            k_idx_his.append(np.argmax(r_list))
            # record K in case of adaptive K
            K_his.append(K)
            mode_his.append(m_list[np.argmax(r_list)])
        except:
            print(i)
            continue
    total_time = time.time() - start_time
    mem.plot_cost()
   # plot_rate(rate_his_ratio)

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
    AoI_t=[1,1,1]
    BEnergy_t = [0.0002,0.0002,0.0002]
    pl_AoI=[]
    Amax = 4
    Bmax = 0.0003
    sigma = 3.162277660168375 * 10 ** (-13)
    S = 15
    theta = []  # never used 权重
    eta = 0.5  # gain loss
    P = 5.012
    FinalAoI=0
    number=0

    for i in range(3000):
        AverSumAoI = 0
        h_t = channel_h[i, :]
        g_t = channel_g[i, :]

        B_test=[x*1000 for x in BEnergy_t]
        #predict = mem.model(torch.Tensor(np.hstack((h_t,g_t,B_test,AoI_t))))
        m_list1=[]
        m_list1 = mem.decode(h_t,g_t,B_test,AoI_t, N, decoder_mode)
        r_list1 = []

        for m in m_list1:
            assert (m[np.argmax(m)] == 1)
            r_list1.append(bisection(h_t / 10000, g_t / 10000, BEnergy_t, AoI_t, m)[0])
        # encode the mode with largest reward

        LyaDrift,AverSumAoI,B_tt,AoI_t=(bisection(h_t / 10000, g_t / 10000, BEnergy_t, AoI, m_list1[np.argmax(r_list1)]))
        BEnergy_t = [x  for x in B_tt]
        '''
        flat = 0
        EnergyHarvest = [0 for j in range(N)]  # amount of energy harvest
        maxj=int(torch.Tensor.argmax(predict))
        for j in range(N+1):
            if maxj==0:
                for j in range(N):
                    EnergyHarvest[j] = eta * P * g_t[j]/10000
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
                    EnergyTrans = sigma / (h_t[j-1]/10000) * (2 ** S)
                    if EnergyTrans > BEnergy_t[j-1]:
                        number+=1
                        flat = 1
                        for k in range(N):
                            EnergyHarvest[k] = eta * P * g_t[k] / 10000
                        for k in range(N):
                            if AoI_t[k] < Amax:
                                AoI_t[k] += 1
                        for k in range(N):
                            if EnergyHarvest[k] + BEnergy_t[k] < Bmax:
                                BEnergy_t[k] += EnergyHarvest[k]
                            else:
                                BEnergy_t[k] = Bmax
                    if flat==0:
                        BEnergy_t -= EnergyTrans
                        AoI_t[j - 1] = 1
                        for k in range(N):
                            if k != j-1 and AoI_t[k] < Amax:
                                AoI_t[k] += 1
        for j in range(N):
            AverSumAoI +=AoI_t[j]
        AoI_text.append([x for x in AoI_t])
        AverSumAoI /= N
        '''
        AoI_text.append([x for x in AoI_t])
        FinalAoI = (FinalAoI *i +AverSumAoI)/(i+1)
        pl_AoI.append(FinalAoI)

    print("Aoi:",FinalAoI)
    save_to_txt(AoI_text, "AoI_text")
    print("number",number)
    plot_AoI(pl_AoI)
