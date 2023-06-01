import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import socket
import threading
import struct
import time

class BufferAgent:
    def __init__(self, buffer_ip, buffer_port, local_ip, local_port, state_dim, action_dim, batch_size, max_data_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = DDPG(state_dim, action_dim)
        self.batch_size = batch_size
        self.buffer_ip = buffer_ip
        self.buffer_port = buffer_port
        self.local_ip = local_ip
        self.local_port = local_port
        self.max_data_size = max_data_size
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.bind((self.local_ip, local_port))
        self.client_socket.connect((self.buffer_ip, self.buffer_port))

    def get_length(self):
        self.client_socket.send(self.int_2_bytes(5)+self.int_2_bytes(0))
        length = self.client_socket.recv(1024)
        return self.bytes_2_int(length)

    def reset(self):
        self.client_socket.send(self.int_2_bytes(2)+self.int_2_bytes(0))

    def save_buffer(self):
        self.client_socket.send(self.int_2_bytes(3)+self.int_2_bytes(0))
        print('save buffer')

    def load_buffer(self):
        self.client_socket.send(self.int_2_bytes(4)+self.int_2_bytes(0))

    def save_model(self):
        self.client_socket.send(self.int_2_bytes(10)+self.int_2_bytes(0))

    def load_model(self):
        self.client_socket.send(self.int_2_bytes(11)+self.int_2_bytes(0))

    def push(self, state, action, reward, next_state):
        # state,
        # print('push data')
        bytes_array = self.int_2_bytes(0)
        bytes_array += self.int_2_bytes(8*self.state_dim + 8*self.action_dim + 8 +8*self.state_dim)
        bytes_array += self.array_2_bytes(state)
        bytes_array += self.array_2_bytes(action)
        bytes_array += self.float_2_bytes(reward)
        bytes_array += self.array_2_bytes(next_state)
        my_list = self.seg_bytes(bytes_array)
        for k in range(len(my_list)):
            # print('send', my_list[k])
            self.client_socket.send(my_list[k])
            # time.sleep(5)

    def seg_bytes(self, bytes_array):
        temp_list = []
        ind = 0
        while(ind < len(bytes_array)):
            temp_list.append(bytes_array[ind:min(ind+1024, len(bytes_array))])
            ind += 1024
        return temp_list

    def sync_actor(self):
        '''(512, state_dim)
        (512,)
        (256, 512)
        (256,)
        (128, 256)
        (128,)
        (action_dim, 128)
        (action_dim,)'''
        self.client_socket.send(self.int_2_bytes(6)+self.int_2_bytes(0))
        # print('ready to catch', self.max_data_size, 'data')
        temp_data = None
        while True:
            recv_data = self.client_socket.recv(self.max_data_size)
            if temp_data == None:
                temp_data = recv_data
            else:
                temp_data += recv_data
            # print('received actor bytes', len(recv_data), ',', len(temp_data))
            if len(temp_data) == self.actor_len():
                break
        mat_list = self.parse_actor(temp_data)
        i = 0
        for param in self.agent.p_net.parameters():
            # print(mat_list[i].shape)
            param.data = nn.parameter.Parameter(torch.from_numpy(mat_list[i]).float())
            i += 1
        # print('synchronize actor')

    def parse_actor(self, recv_data):
        '''(256, state_dim)
                (256,)
                (128, 256)
                (128,)
                (action_dim, 128)
                (action_dim,)'''
        mat_list = []
        ind = 0
        new_mat = self.bytes_2_mat(recv_data[ind:ind + 8 * 256 * self.state_dim], (256, self.state_dim))
        mat_list.append(new_mat)
        ind += 8 * 256 * self.state_dim

        new_mat = self.bytes_2_array(recv_data[ind:ind + 8 * 256], 256)
        mat_list.append(new_mat)
        ind += 8 * 256

        new_mat = self.bytes_2_mat(recv_data[ind:ind + 8 * 128 * 256], (128, 256))
        mat_list.append(new_mat)
        ind += 8 * 128 * 256

        new_mat = self.bytes_2_array(recv_data[ind:ind + 8 * 128], 128)
        mat_list.append(new_mat)
        ind += 8 * 128

        new_mat = self.bytes_2_mat(recv_data[ind:ind + 8 * self.action_dim * 128], (self.action_dim, 128))
        mat_list.append(new_mat)
        ind += 8 * self.action_dim * 128

        new_mat = self.bytes_2_array(recv_data[ind:ind + 8 * self.action_dim], self.action_dim)
        mat_list.append(new_mat)
        ind += 8 * self.action_dim
        '''print('parse actor')
        for i in range(6):
            print(mat_list[i].shape)'''
        return mat_list

    def actor_len(self):
        a = 0
        for param in self.agent.p_net.parameters():
            mat = param.data.detach().cpu().numpy()
            if mat.ndim == 2:
                a += 8*mat.shape[0]*mat.shape[1]
            elif mat.ndim == 1:
                a += 8*mat.shape[0]
        return a

    def start_train(self):
        self.client_socket.send(self.int_2_bytes(7)+self.int_2_bytes(0))

    def stop_train(self):
        self.client_socket.send(self.int_2_bytes(8)+self.int_2_bytes(0))

    def train_one_step(self):
        self.client_socket.send(self.int_2_bytes(12)+self.int_2_bytes(0))

    def int_2_bytes(self, x):
        # 4 bytes
        return struct.pack(">I", x)

    def bytes_2_int(self, x):
        return struct.unpack('>I', x)[0]

    def float_2_bytes(self, x):
        # 8 bytes
        return struct.pack('d', x)

    def bytes_2_float(self, x):
        return struct.unpack('d', x)[0]

    def array_2_bytes(self, state):
        bytes_array = None
        for i in range(state.shape[0]):
            if bytes_array == None:
                bytes_array = self.float_2_bytes(state[0])
            else:
                bytes_array += self.float_2_bytes(state[i])
        return bytes_array

    def bytes_2_array(self, bytes_array, dim):
        # print(bytes_array)
        trans_mat = np.zeros((dim,))
        for i in range(dim):
            trans_mat[i] = self.bytes_2_float(bytes_array[8 * i:8 * i + 8])
        return trans_mat

    def bytes_2_mat(self, bytes_array, dim):
        trans_mat = np.zeros(dim)
        for i in range(dim[0]):
            for j in range(dim[1]):
                trans_mat[i, j] = self.bytes_2_float(bytes_array[dim[1] * 8 * i + 8 * j:dim[1] * 8 * i + 8 * j + 8])
        return trans_mat

    def mat_2_bytes(self, hello):
        bytes_array = None
        for i in range(hello.shape[0]):
            for j in range(hello.shape[1]):
                if bytes_array == None:
                    bytes_array = self.float_2_bytes(hello[0][0])
                else:
                    bytes_array += self.float_2_bytes(hello[i][j])
        return bytes_array

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x * 2   # only for this environment
        return x

class DDPG(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_1_net = QNetwork(state_dim, action_dim)
        self.q_2_net = QNetwork(state_dim, action_dim)
        self.target_q_1_net = QNetwork(state_dim, action_dim)
        self.target_q_2_net = QNetwork(state_dim, action_dim)
        for target_param, param in zip(self.target_q_1_net.parameters(), self.q_1_net.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_q_2_net.parameters(), self.q_2_net.parameters()):
            target_param.data.copy_(param.data)
        self.soft_tau = 1e-2

        self.p_net = PolicyNetwork(state_dim, action_dim)
        self.q_criterion = nn.MSELoss()
        self.q_1_optimizer = optim.Adam(self.q_1_net.parameters(), lr=1e-3)
        self.q_2_optimizer = optim.Adam(self.q_2_net.parameters(), lr=1e-3)
        self.p_optimizer = optim.Adam(self.p_net.parameters(), lr=3e-4)
        self.gamma = 0.99

    def get_action(self, state, epsilon):
        a = self.p_net.forward(torch.from_numpy(state).float())
        a = a + epsilon * torch.randn(self.action_dim)
        a = torch.clamp(a, min=-2, max=2)
        return a.detach().numpy()

    def train(self, batch):
        state = batch[0]    # array [64 1 2]
        action = batch[1]   # array [64, ]
        reward = batch[2]   # array [64, ]
        next_state = batch[3]

        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float().view(-1, self.action_dim)
        next_state = torch.from_numpy(next_state).float()
        next_action = self.p_net.forward(next_state)
        reward = torch.FloatTensor(reward).float().unsqueeze(1)

        q1 = self.q_1_net.forward(state, action)
        q2 = self.q_2_net.forward(state, action)
        # q_min = torch.min(q1, q2)
        next_q1 = self.target_q_1_net.forward(next_state, next_action)
        next_q2 = self.target_q_2_net.forward(next_state, next_action)
        next_q_min = torch.min(next_q1, next_q2)
        est_q = reward + self.gamma * next_q_min

        q_loss = self.q_criterion(q1, est_q.detach())
        self.q_1_optimizer.zero_grad()
        q_loss.backward()
        self.q_1_optimizer.step()
        q_loss = self.q_criterion(q2, est_q.detach())
        self.q_2_optimizer.zero_grad()
        q_loss.backward()
        self.q_2_optimizer.step()

        new_a = self.p_net.forward(state)
        q1 = self.q_1_net.forward(state, new_a)
        q2 = self.q_2_net.forward(state, new_a)
        q_min = torch.min(q2, q1)
        p_loss = -q_min.mean()
        self.p_optimizer.zero_grad()
        p_loss.backward()
        self.p_optimizer.step()

        for target_param, param in zip(self.target_q_1_net.parameters(), self.q_1_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
        for target_param, param in zip(self.target_q_2_net.parameters(), self.q_2_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def load_model(self):
        print('load model')
        self.q_1_net = torch.load('model/DDPG_q_1_net.pkl')
        self.q_2_net = torch.load('model/DDPG_q_2_net.pkl')
        self.target_q_1_net = torch.load('model/DDPG_target_q_1_net.pkl')
        self.target_q_2_net = torch.load('model/DDPG_target_q_2_net.pkl')
        self.p_net = torch.load('model/DDPG_policy_net.pkl')

    def save_model(self):
        torch.save(self.q_1_net, 'model/DDPG_q_1_net.pkl')
        torch.save(self.q_2_net, 'model/DDPG_q_2_net.pkl')
        torch.save(self.target_q_1_net, 'model/DDPG_target_q_1_net.pkl')
        torch.save(self.target_q_2_net, 'model/DDPG_target_q_2_net.pkl')
        torch.save(self.p_net, 'model/DDPG_policy_net.pkl')

if __name__ == '__main__':
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    state_dim = 3
    action_dim = 1
    max_epi_iter = 200
    max_MC_iter = 200
    batch_size = 64
    local_ip = "192.168.1.132"
    local_port = 6670
    max_data_size = 2000000  # 1427512
    buffer_agent = BufferAgent("192.168.1.132", 6666, local_ip, local_port, state_dim, action_dim, batch_size,
                               max_data_size)
    # buffer_agent.reset()
    train_curve = []
    # buffer_agent.stop_train()
    for epi in range(max_epi_iter):
        state = env.reset()[0]
        acc_reward = 0
        for MC_iter in range(max_MC_iter):
            # print("MC= ", MC_iter)
            env.render()
            action1 = buffer_agent.agent.get_action(state, 1.0-(epi/max_epi_iter))
            # print(state.shape, action1.shape)   # (3,) (1,)
            next_state, reward, done, info, _ = env.step(action1)
            acc_reward = acc_reward + reward
            buffer_agent.push(state.reshape((state_dim,)), action1.reshape((action_dim,)), reward, next_state.reshape((state_dim,)))
            # print('epi', epi, 'MC', MC_iter, 'save buffer', state, action1, reward, next_state)
            state = next_state
            if done:
                break
        print('Episode', epi, 'reward', acc_reward)
        train_curve.append(acc_reward)
        buffer_agent.sync_actor()
        time.sleep(2.0)
        np.save('data/remote_con_train_sync_each_epi_sleep_2_1.npy', np.array(train_curve))
    # plt.plot(train_curve, linewidth=1, label='DDPG')
    # plt.show()

    '''buffer_agent.stop_train()
    # buffer_agent.start_train()
    buffer_agent.reset()
    bytes_array = buffer_agent.int_2_bytes(0)
    bytes_array += buffer_agent.int_2_bytes(8 * buffer_agent.state_dim + 8 * buffer_agent.action_dim + 8 + 8 * buffer_agent.state_dim)
    buffer_agent.client_socket.send(bytes_array)
    time.sleep(2)

    state = np.zeros((state_dim, )) + 0.0
    action = np.zeros((action_dim, )) + 1.0
    reward = 2.0
    next_state = np.zeros((state_dim, )) + 3.0

    bytes_array = buffer_agent.array_2_bytes(state)
    buffer_agent.client_socket.send(bytes_array)
    time.sleep(2)
    bytes_array = buffer_agent.array_2_bytes(action)
    buffer_agent.client_socket.send(bytes_array)
    time.sleep(2)
    bytes_array = buffer_agent.float_2_bytes(reward)
    buffer_agent.client_socket.send(bytes_array)
    time.sleep(2)
    bytes_array = buffer_agent.array_2_bytes(next_state)
    buffer_agent.client_socket.send(bytes_array)
    time.sleep(2)

    buffer_agent.save_buffer()
    buffer_agent.load_buffer()
    print('buffer length', buffer_agent.get_length())
    buffer_agent.sync_actor()
    buffer_agent.save_model()
    buffer_agent.load_model()

    while(True):
        pass'''