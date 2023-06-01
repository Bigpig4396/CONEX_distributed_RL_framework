import socket
import threading
import struct
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

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

class ReplayBuffer:
    def __init__(self, capacity, local_ip, local_port, max_data_size, state_dim, action_dim, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.capacity = capacity
        self.max_data_size = max_data_size
        self.ip = local_ip # socket.gethostbyname(socket.gethostname())
        self.local_port = local_port
        print('initialize buffer at', self.ip, ', port', local_port)

        self.is_train_mode = 1
        self.buffer = []
        self.position = 0
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
        self.server_socket.bind((self.ip, local_port))
        self.server_socket.listen(64)        # max 64 workers
        self.lock = threading.Lock()
        self.agent = DDPG(state_dim, action_dim)
        self.run()

    def train_model(self):
        i = 0
        while True:
            if self.is_train_mode == 1:
                if len(self.buffer) > self.batch_size:
                    if i % 100 == 0:
                        print('train iter', i, 'buffer size', len(self.buffer))
                    self.lock.acquire()
                    batch = self.sample(self.batch_size)
                    self.agent.train(batch)
                    self.lock.release()
                    time.sleep(0.01)
                    i += 1
                else:
                    self.lock.acquire()
                    print('buffer not available', len(self.buffer))
                    self.lock.release()
                    time.sleep(2)
            else:
                time.sleep(2)

    def run(self):
        thd = threading.Thread(target=self.train_model)
        thd.setDaemon(True)
        thd.start()
        while True:
            client_socket, client_addr = self.server_socket.accept()
            print('accept new connection', client_addr)
            thd = threading.Thread(target=self.on_client, args=(client_socket, client_addr))
            thd.setDaemon(True)
            thd.start()

    def on_client(self, client_socket, client_addr):
        msg_buffer = None
        while True:
            recv_data = client_socket.recv(self.max_data_size)
            if recv_data:
                if msg_buffer == None:
                    msg_buffer = recv_data
                else:
                    msg_buffer += recv_data

                pos = 0
                rest_len = len(msg_buffer) - pos
                # print('receive data', len(recv_data), 'message buffer length', len(msg_buffer), 'rest_len', rest_len)
                while rest_len >=8:
                    # print('decoding', len(msg_buffer))
                    inst = self.bytes_2_int(msg_buffer[pos:pos+4])
                    exp_msg_len = self.bytes_2_int(msg_buffer[pos+4:pos+8])

                    # print('inst', inst)
                    # print('exp_msg_len', exp_msg_len)
                    if rest_len < exp_msg_len + 8:
                        msg_buffer = msg_buffer[pos:]
                        pos = 0
                        # print('inst not finish')
                        break
                    if inst == 0:
                        # print('instruction: push data')
                        self.lock.acquire()
                        pos  += 8
                        state = self.bytes_2_array(msg_buffer[pos:pos + 8 * self.state_dim], self.state_dim)
                        pos += 8 * self.state_dim
                        action = self.bytes_2_array(msg_buffer[pos:pos + 8 * self.action_dim], self.action_dim)
                        pos += 8 * self.action_dim
                        reward = self.bytes_2_float(msg_buffer[pos:pos + 8])
                        pos += 8
                        next_state = self.bytes_2_array(msg_buffer[pos:pos + 8 * self.state_dim], self.state_dim)
                        pos += 8 * self.state_dim
                        # print('state', state)
                        # print('action', action)
                        # print('reward', reward)
                        # print('next_state', next_state)
                        self.push(state, action, reward, next_state)
                        self.lock.release()
                    elif inst == 1:
                        # print('instruction: sample data')
                        self.lock.acquire()
                        if len(self.buffer) < self.batch_size:
                            bytes_array = self.int_2_bytes(0)   # buffer not available
                        else:
                            bytes_array = self.int_2_bytes(1)   # buffer is available
                            state, action, reward, next_state = self.sample(self.batch_size)
                            for k in range(self.batch_size):
                                bytes_array += self.array_2_bytes(state[k, :])
                                bytes_array += self.array_2_bytes(action[k, :])
                                bytes_array += self.float_2_bytes(reward[k])
                                bytes_array += self.array_2_bytes(next_state[k, :])
                        client_socket.send(bytes_array)
                        pos += 8
                        self.lock.release()
                    elif inst == 2:
                        # print('instruction: reset buffer')
                        self.lock.acquire()
                        self.reset()
                        pos += 8
                        self.lock.release()
                    elif inst == 3:
                        # print('instruction: save buffer')
                        self.lock.acquire()
                        self.save()
                        pos += 8
                        self.lock.release()
                        # self.print_data()
                    elif inst == 4:
                        # print('instruction: load buffer')
                        self.lock.acquire()
                        self.load()
                        pos += 8
                        self.lock.release()
                    elif inst == 5:
                        # print('instruction: get buffer length')
                        self.lock.acquire()
                        client_socket.send(self.int_2_bytes(len(self.buffer)))
                        pos += 8
                        self.lock.release()
                    elif inst == 6:
                        self.lock.acquire()
                        bytes_array = self.actor_2_bytes()
                        # print('instruction: synchronize actor', len(bytes_array))
                        my_list = self.seg_actor_bytes(bytes_array)
                        for k in range(len(my_list)):
                            client_socket.send(my_list[k])
                        # print('send actor data finish')
                        pos += 8
                        self.lock.release()
                    elif inst == 7:
                        # print('instruction: start training')
                        self.lock.acquire()
                        self.is_train_mode = 1
                        pos += 8
                        self.lock.release()
                    elif inst == 8:
                        # print('instruction: stop training')
                        self.lock.acquire()
                        self.is_train_mode = 0
                        pos += 8
                        self.lock.release()
                    elif inst == 9:
                        # print('instruction: set zero actor')
                        self.lock.acquire()
                        # self.set_zero_actor()
                        pos += 8
                        self.lock.release()
                    elif inst == 10:
                        # print('instruction: save model')
                        self.lock.acquire()
                        self.agent.save_model()
                        pos += 8
                        self.lock.release()
                    elif inst == 11:
                        # print('instruction: load model')
                        self.lock.acquire()
                        self.agent.load_model()
                        pos += 8
                        self.lock.release()
                    elif inst == 12:
                        # print('instruction: train one step')
                        self.lock.acquire()
                        batch = self.sample(self.batch_size)
                        self.agent.train(batch)
                        pos += 8
                        self.lock.release()
                    msg_buffer = msg_buffer[pos:]
                    pos = 0
                    rest_len = len(msg_buffer) - pos

            else:
                print('client', client_addr, 'disconnect')
                break
        client_socket.close()
        print('close client socket', client_addr)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def reset(self):
        self.buffer = []
        self.position = 0

    def save(self):
        print('save data')
        # print(self.buffer)
        if len(self.buffer) == 0:
            print('there is no data in the buffer')
        else:
            state, action, reward, next_state = map(np.stack, zip(*(self.buffer)))
            np.save('data/DB_state.npy', state)
            np.save('data/DB_action.npy', action)
            np.save('data/DB_reward.npy', reward)
            np.save('data/DB_next_state.npy', next_state)
            np.save('data/DB_position.npy', np.array([self.position]))
        print(' ')

    def load(self):
        print('load data')
        temp_state = np.load('data/DB_state.npy')
        temp_action = np.load('data/DB_action.npy')
        temp_reward = np.load('data/DB_reward.npy')
        temp_next_state = np.load('data/DB_next_state.npy')
        self.buffer = []
        self.position = 0
        for i in range(temp_state.shape[0]):
            # print(temp_state[i, :], temp_action[i, :], temp_reward[i], temp_next_state[i, :])
            self.push(temp_state[i, :], temp_action[i, :], temp_reward[i], temp_next_state[i, :])
        self.position = int(np.load('data/DB_position.npy'))
        print(' ')

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity
        # print('save buffer', state, action, reward, next_state)

    def actor_2_bytes(self):
        bytes_array = None
        for param in self.agent.p_net.parameters():
            mat = param.data.detach().cpu().numpy()
            # print(mat)
            if mat.ndim == 2:
                if bytes_array == None:
                    bytes_array = self.mat_2_bytes(mat)
                else:
                    bytes_array += self.mat_2_bytes(mat)
            elif mat.ndim == 1:
                if bytes_array == None:
                    bytes_array = self.array_2_bytes(mat)
                else:
                    bytes_array += self.array_2_bytes(mat)
        return bytes_array

    def seg_actor_bytes(self, bytes_array):
        temp_list = []
        ind = 0
        while(ind < len(bytes_array)):
            temp_list.append(bytes_array[ind:min(ind+1024, len(bytes_array))])
            ind += 1024
        return temp_list

    def bytes_2_int(self, x):
        return struct.unpack('>I', x)[0]

    def array_2_bytes(self, state):
        bytes_array = None
        for i in range(state.shape[0]):
            if bytes_array == None:
                bytes_array = self.float_2_bytes(state[0])
            else:
                bytes_array += self.float_2_bytes(state[i])
        return bytes_array

    def float_2_bytes(self, x):
        # 8 bytes
        return struct.pack('d', x)

    def bytes_2_array(self, bytes_array, dim):
        # print(bytes_array)
        trans_mat = np.zeros((dim,))
        for i in range(dim):
            trans_mat[i] = self.bytes_2_float(bytes_array[8 * i:8 * i + 8])
        return trans_mat

    def bytes_2_float(self, x):
        return struct.unpack('d', x)[0]

    def int_2_bytes(self, x):
        # 4 bytes
        return struct.pack(">I", x)

    def mat_2_bytes(self, hello):
        bytes_array = None
        for i in range(hello.shape[0]):
            for j in range(hello.shape[1]):
                if bytes_array == None:
                    bytes_array = self.float_2_bytes(hello[0][0])
                else:
                    bytes_array += self.float_2_bytes(hello[i][j])
        return bytes_array

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim = 3
    action_dim = 1
    batch_size = 64
    max_data_size = 65536  # may change depending on your state/action size
    local_ip = "192.168.1.132"
    local_port = 6666
    replay_buffer = ReplayBuffer(1000000, local_ip, local_port, max_data_size, state_dim, action_dim, batch_size)