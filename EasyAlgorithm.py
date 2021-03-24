import scipy.io as sio
import numpy as np

from mainPyTorch import save_to_txt

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
    plt.ylabel('Averagy Sum AoI')
    plt.xlabel('Time Frames')
    plt.show()
if __name__ == "__main__":
    N = 5
    n=100 #time slots
    data = sio.loadmat('./data/data_%d' %N)
    channel_h = data['input_h']
    channel_g = sio.loadmat('./data/data_%d' %N)['input_g']
    NodeBEnergy = sio.loadmat('./data/data_%d' % N)['input_battery']
    BEnergy = NodeBEnergy[1, :]
    AoI = [1, 1, 1, 1, 1]
    V = 1   # Lyapnov drift value
    flat = 1  # define H or T
    Amax = 6
    Bmax = 0.0004
    sigma = 3.162277660168375 * 10 ** (-13)
    S = 12
    FinalAoI = 0
    theta = []  # never used 权重
    eta = 0.5  # gain loss
    P = 5.012
    LyaBEnergy = 0  # calculate Battery Energy changed
    index = []
    AoI_index=[]
    BEnergy_index=[]
    for i in range(n):
        AverSumAoI = 0  # Sum of AoI at base station
        EnergyTrans = [0 for j in range(N)]
        EnergyHarvest = [0 for j in range(N)]  # amount of energy harvest
        h = channel_h[n, :]
        g = channel_g[n, :]
        for j in range(N):
            EnergyHarvest[j] = eta * P * g[j]
            EnergyTrans[j] = sigma/h[j]*(2**S)
        flat = 0 #标志有没有节点能传输数据
        nodeToTrans = 0 #哪个节点传输数据
        for j in range(N):
            for k in range(N):
                if(h[j]>h[k] and EnergyTrans[j]<=BEnergy[j]):
                    flat = 1
                    nodeToTrans = j
        if(flat == 0):  #吸收能量
            for j in range(N):
                if(AoI[j]!=Amax):
                    AoI[j] += 1
                if(BEnergy[j]+EnergyHarvest[j]>=Bmax):
                    BEnergy[j] = Bmax
                else:
                    BEnergy[j] += EnergyHarvest[j]
        else:  #nodetoEnergy传输能量
            BEnergy[nodeToTrans] -= EnergyTrans[nodeToTrans]
            AoI[nodeToTrans] = 1
            for j in range(N):
                if j != nodeToTrans and AoI[j]!= Amax:
                    AoI[j] += 1
        for j in range(N):
            AverSumAoI += AoI[j]
        AverSumAoI /= N
        FinalAoI = (FinalAoI*i +AverSumAoI)/(i+1)
        index.append(FinalAoI)
        AoI_index.append([i for i in AoI])
        BEnergy_index.append([i for i in BEnergy])
        save_to_txt(index,"index")
        save_to_txt(AoI_index,"AoI_index")
        save_to_txt(BEnergy_index,"BEnergy_index")
        plot_rate(index,20)
    print(FinalAoI)
