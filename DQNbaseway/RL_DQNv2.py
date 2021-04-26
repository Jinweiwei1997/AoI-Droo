# 用nn为DQN构造网络
import numpy as np
import torch
import torch.nn as nn

from DQNbaseway.mazeEnv import Maze
from mainPyTorch import save_to_txt


# import


class DQN():
    def __init__(self,
                 dim_state,
                 n_actions,
                 batch_size=32,
                 learning_rate=0.001,
                 epsilon=0.9,
                 gamma=0.9,
                 training_interval=10,
                 target_replace_iter=100,
                 memory_size=2000, ):
        # 调用类内自写函数生成网络
        self.eval_net, self.target_net = self.bulid_Net(dim_state, n_actions), self.bulid_Net(dim_state, n_actions)

        self.dim_state = dim_state  # 状态维度
        self.n_actions = n_actions  # 可选动作数
        self.batch_size = batch_size        # 小批量梯度下降，每个“批”的size
        self.learning_rate = learning_rate  # 学习率
        self.epsilon = epsilon  # 贪婪系数
        self.gamma = gamma      # 回报衰减率
        self.memory_size = memory_size                  # 记忆库的规格
        self.taget_replace_iter = target_replace_iter   # target网络延迟更新的间隔步数
        self.learn_step_counter = 0     # 在计算隔n步跟新的的时候用到
        self.memory_counter = 0         # 用来计算存储索引
        self.memory = np.zeros((self.memory_size, self.dim_state * 2 + 2))  # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)    # 网络优化器
        self.loss_func = nn.MSELoss()   # 网络的损失函数
        self.cost_his = []
        self.training_interval=training_interval
        self.a=0.8
        self.fixState=np.array([1,2,1,1,1,2,0,1,2,2,2,1])

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()
    # 选择动作
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.epsilon:  # greedy概率有eval网络生成动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1]
            action = int(action)
        else:  # （1-greedy）概率随机选择动作
            action = np.random.randint(0, self.n_actions)
        return action

    def final_choose(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1]
        action = int(action)
        return action
    # 学习，更新网络参数
    def learn(self):
        # 目标网络参数更新（经过self.taget_replace_iter步之后，为target_net网络更新参数）
        if self.learn_step_counter % self.taget_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 从记忆库中提取一个batch的数据
        data_size = self.memory_size if self.memory_counter>self.memory_size else self.memory_counter

        sample_index = np.random.choice(data_size, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.dim_state])
        b_a = torch.LongTensor(b_memory[:, self.dim_state:self.dim_state + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.dim_state + 1:self.dim_state + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.dim_state:])
        f_sa=[]
        for i in range(32):
            f_sa.append(self.fixState)
        f_s = torch.FloatTensor(f_sa)
        # 获得q_eval、q_target，计算loss
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_fix =self.target_net(f_s).detach()
        q_target =q_eval  + self.a *(b_r+ q_next.max(1)[0].view(self.batch_size, 1)-q_fix.max(1)[0].view(self.batch_size, 1)-q_eval)
        loss = self.loss_func(q_eval, q_target)


        # 反向传递，更新eval网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cost = loss.item()
        self.cost_his.append(self.cost)
    # 存储一步的信息到记忆库
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 存储记忆（如果第一轮存满了，就覆盖存入）
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    # 构建网络
    def bulid_Net(self, dim_state, n_actions):
        return torch.nn.Sequential(
            torch.nn.Linear(dim_state, 80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 110),
            torch.nn.ReLU(),
            torch.nn.Linear(110, n_actions),
        )
def plot_rate(rate_his, rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    mpl.style.use('seaborn')
    plt.plot(np.arange(len(rate_array))+1, rate_his)
    plt.ylabel('Averagy Sum AoI')
    plt.xlabel('Time Frames')
    plt.show()
if __name__ == '__main__':
    env = Maze()
    dqn = DQN(env.n_states, env.n_actions)
    #print(dqn)

    print('Collecting experience...')
    for i_episode in range(200):
        s = env.reset()                 # 重置初始状态
        ep_r = 0
        for i in range(300):
            #env.render()                # 刷新画面
            aoi=[]
            a = dqn.choose_action(s)    # 选择动作
            s_, r, done,aoi = env.step(a)   # 执行动作，获得下一个状态s_，回报r，是否结束标记done

            if done == False:                    # 如果done（智能到达终点/掉入陷阱），结束本轮
               aoi=[]
            #   a=0
               s_, r1, done, aoi = env.step(0)  # 执行动作，获得下一个状态s_，回报r，是否结束标记done
            dqn.store_transition(s, a, r, s_)   # 存储 一步 的信息
            if dqn.memory_counter > dqn.memory_size:    # 当记忆库存满（非必要等到存满）的时候，开始训练
                dqn.learn()
            s = s_
    save_to_txt(dqn.cost_his,"dqn_cost")
    dqn.plot_cost();
    '''
    # 测试部分
    print('Testing . . .')
    # dqn.epsilon = 1
    rs = []
    for state in range(50): # 打算循环测试50次测一测平均回报
        s = env.reset()
        ep_r = 0
        while True:
            #env.render()
            a = dqn.choose_action(s)
            s_, r, done = env.step(a)
            ep_r += r
            # 测试阶段就不再有存储和学习了
            if done:
                print(ep_r)
                rs.append(ep_r)
                break
            s = s_

    env.close()

    print(np.average(rs))
    '''
    print("测试部分")
    s1=np.array([0, 2, 2, 1,2,3,2, 1,3, 0,2,1])
    s=s1
    dqn_aoi =[]
    dqn_AverAoI=0
    x=0
    env.state=s
    dqn_LongTime=[]
    for i_episode in range(3000):
        ep_r = 0
        aoi=[]
        a = dqn.final_choose(s)    # 选择动作
        s_, r, done,aoi = env.step(a,i_episode+1,"test")   # 执行动作，获得下一个状态s_，回报r，是否结束标记done
        if(done==False):
            aoi=[]
            s_, r, done, aoi = env.step(0)
            x+=1
        dqn_aoi.append(aoi)
        dqn_AverAoI = (dqn_AverAoI*i_episode+r)/(i_episode+1)
        dqn_LongTime.append(-dqn_AverAoI)
        s=s_
    print(-dqn_AverAoI)
    print(x)
    save_to_txt(dqn_aoi, "dqn_aoi")
    save_to_txt(dqn_LongTime,"dqn_LongTime")
    plot_rate(dqn_LongTime)
