import numpy as np
import pandas as pd
import time

N_STATES = 6  # 一维世界的宽度
ACTIONS = ['left','right']  # 探索者的可用动作
EPSTLON = 0.9  # 贪婪度greedy
ALPHA = 0.1  # 学习率
GAMMA = 0.9  # 奖励递减值
MAX_EPISODES = 13  # 最大回合数
FRESH_TIME = 0.3  # 移动间隔时间


# 创建Q表
def build_q_table(n_states,actions):
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))),
        columns = actions  # columns 对应的是行为名称
        )
    return table

#  定义动作
def choose_action(state,q_table):
    state_actions = q_table.iloc[state,:]  # 选出这个state的所有action值
    if (np.random.uniform() > EPSTLON) or (state_actions.all()==0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name

# 环境反馈S_,R
def get_env_feedback(S,A):
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R

# 环境更新

def update_env(S,episode,step_counter):
    env_list = ['-']*(N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1,step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r           ',end='')
    else:
        env_list[S]='o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATES,ACTIONS)  # 初始化q table
    for episode in range(MAX_EPISODES):  # 回合
        step_counter = 0
        S = 0  # 回合初始位置
        is_terminated = False # 是否回合结束
        update_env(S,episode,step_counter)  # 环境更新
        while not is_terminated:
            A = choose_action(S,q_table)  # 选择行为
            S_,R = get_env_feedback(S,A)   # 实施行为并得到环境的反馈
            q_predict = q_table.ix[S,A]    # 估算的（状态-行为）值
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_,:].max()  # 实际的状态-行为值（回合没结束）
            else:
                q_target = R # 实际的状态-行为值（回合结束）
                is_terminated = True
            q_table.ix[S,A] += ALPHA * (q_target - q_predict) # q_table更新
            S = S_  # 移动到下一个state
            update_env(S,episode,step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)


