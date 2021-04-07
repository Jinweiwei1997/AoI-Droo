from memoryPyTorch import MemoryDNN


def bisection(h,g,BEnergy,AoI,M):
    #AoISum The average sum of processes at base station
    V=1 #Lyapnov drift value
    flat=1 #define H or T
    AoI_k=[x for x in AoI] #k slot AoI
    Amax=4
    Bmax=0.0003
    sigma=3.162277660168375*10**(-13)
    S=15
    AverSumAoI = 0 #Sum of AoI at base station
    theta =[] #never used 权重
    eta = 0.5 #gain loss
    P = 5.012

    EnergyHarvest = [0 for i in range(len(M)-1)] #amount of energy harvest
    BEnergy_k = [x for x in BEnergy]
    LyaBEnergy = 0 #calculate Battery Energy changed
    B_Lya2 = 0
    for i in range(len(M)):
        if M[0] == 1:
            # the model is H
            for j in range(len(M) - 1):
                # calculate the Energy Harvested
                EnergyHarvest[j] = eta * P * g[j]
                B_next = BEnergy_k[j] + EnergyHarvest[j]
                if B_next >= Bmax:
                    BEnergy_k[j] = Bmax
                else:
                    BEnergy_k[j] += EnergyHarvest[j]
                # calculate the Sum of AoI
            for j in range(len(M) - 1):
                if (AoI[j] < Amax):
                    AoI_k[j] = AoI[j] + 1
                else:
                    AoI_k[j] = Amax
            for j in range(len(M) - 1):
                AverSumAoI += AoI_k[j]
            AverSumAoI /= (len(M) - 1)
            break
        else:
            if(M[i]==1 ):
                for j in range(len(M) - 1):
                    if ((i - 1) != j):
                        if AoI_k[j] < Amax:
                            AoI_k[j] = AoI[j] + 1
                        else:
                            AoI_k[j] = Amax
                    else:
                        AoI_k[j] = 1
                # calcuate Sum of AoI
                for j in range(len(M) - 1):
                    AverSumAoI += AoI_k[j]
                AverSumAoI /= (len(M) - 1)
                # calculate the Energy for transaction
                EnergyTrans = sigma / h[i - 1] * (2 ** S)
                if (BEnergy_k[i - 1] > EnergyTrans):
                    BEnergy_k[i - 1] -= EnergyTrans
                else:
                    return -1000000.01, BEnergy;
                break


    for i in range(len(M)-1):
        LyaBEnergy += (BEnergy_k[i]-BEnergy[i])*(BEnergy_k[i]-BEnergy[i])
        B_Lya2 += BEnergy_k[i]*BEnergy_k[i]
    LyapnovDrift = -LyaBEnergy - AverSumAoI
    return LyapnovDrift,AverSumAoI,BEnergy_k,AoI_k


'''
if __name__ == "__main__":
    h =([6.06020304235508 * 10 ** -6, 1.10331933767028 * 10 ** -5, 1.00213540309998 * 10 ** -7,1.21610610942759 * 10 ** -6])
    g =([6.06020304235508 * 10 ** -6, 1.10331933767028 * 10 ** -5, 1.00213540309998 * 10 ** -7,1.21610610942759 * 10 ** -6])
    M = ([0, 0, 0, 0])
    BEnergy =[4*10**-4,4*10**-4,4*10**-4,4*10**-4]
    AoI=([1,1,1,1])
    a = bisection(h,g,BEnergy,AoI,M)
    print(a)
'''
if __name__ == "__main__":
    N=10;Memory = 1024
    mem = MemoryDNN(net = [N, 120, 80, N],
                    learning_rate = 0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory
                    )
    mlist=mem.decode(mem,5)
    print(mlist)