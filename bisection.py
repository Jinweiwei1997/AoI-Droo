from memoryPyTorch import MemoryDNN


def bisection(h,g,BEnergy,AoI,M):
    #AoISum The average sum of processes at base station
    V=1 #Lyapnov drift value
    flat=1 #define H or T
    AoI_k=[x for x in AoI] #k slot AoI
    Amax=6
    Bmax=0.0004
    sigma=3.162277660168375*10**(-13)
    S=12
    AverSumAoI = 0 #Sum of AoI at base station
    theta =[] #never used 权重
    eta = 0.5 #gain loss
    P = 5.012
    EnergyHarvest = [0 for i in range(len(M))] #amount of energy harvest
    BEnergy_k = [x for x in BEnergy]
    LyaBEnergy = 0 #calculate Battery Energy changed
    for i in range(len(M)):
        if (M[i]==1):
            #the model is T
            flat=0
            for j in range(len(M)):
                if(i!=j):
                    if AoI_k[j] < Amax:
                        AoI_k[j] = AoI[j]+1
                    else:
                        AoI_k[j] = Amax
                else:
                    AoI_k[j]=1
            #calcuate Sum of AoI
            for j in range(len(M)):
                AverSumAoI += AoI_k[j]
            AverSumAoI /= len(M)
            #calculate the Energy for transaction
            EnergyTrans = sigma/h[i]*(2**S)
            if(BEnergy_k[i]>EnergyTrans):
                BEnergy_k[i] -=EnergyTrans
            else:
                return 1000000;
            break
    if flat==1:
        #the model is H
        for i in range(len(M)):
            #calculate the Energy Harvested
            EnergyHarvest[i]= eta*P*g[i]
            B_next=BEnergy_k[i] + EnergyHarvest[i]
            if B_next>=Bmax:
                BEnergy_k[i]=Bmax
            else:
                BEnergy_k[i] += EnergyHarvest[i]
            #calculate the Sum of AoI
        for i in range(len(M)):
            if(AoI[i] < Amax):
                AoI_k[i] = AoI[i]+1
            else:
                AoI_k[i]=Amax
        for i in range(len(M)):
            AverSumAoI += AoI_k[i]
        AverSumAoI /= len(M)
    for i in range(len(M)):
        LyaBEnergy += (BEnergy_k[i]-BEnergy[i])*(BEnergy_k[i]-BEnergy[i])+AverSumAoI
    LyapnovDrift = LyaBEnergy + AverSumAoI
    return LyapnovDrift,BEnergy_k,AoI_k


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