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
from torch.distributions import Normal

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = -20
        self.log_std_max = 2


        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mean = nn.Linear(128, self.action_dim)
        self.log_std = nn.Linear(128, self.action_dim)

    def sample_action(self, ):
        a = torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return self.action_range * a.numpy()

    def forward(self, state):
        h1 = F.relu(self.fc1(state))
        x = F.relu(self.fc2(h1))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        z = Normal(0, 1).sample(mean.shape)
        action_0 = torch.tanh(mean + std * z)
        action = action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1. - action_0.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        z = Normal(0, 1).sample(mean.shape)
        action = torch.tanh(mean + std * z)
        action = action.detach().numpy()
        return action

class SAC():
    def __init__(self, state_dim, action_dim):
        self.soft_q_net1 = QNetwork(state_dim, action_dim)
        self.soft_q_net2 = QNetwork(state_dim, action_dim)
        self.target_soft_q_net1 = QNetwork(state_dim, action_dim)
        self.target_soft_q_net2 = QNetwork(state_dim, action_dim)
        self.policy_net = Actor(state_dim, action_dim)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        self.gamma = 0.99
        self.soft_tau = 1e-2
        self.reward_scale = 10.0
        self.target_entropy = -1. * action_dim

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=3e-4)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=3e-4)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def get_action(self, state):
        a = self.policy_net.get_action(state)
        return a

    def train(self, batch):
        state, action, reward, next_state = batch

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(-1)

        predicted_q_value1 = self.soft_q_net1.forward(state, action)
        predicted_q_value2 = self.soft_q_net2.forward(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)

        # Updating alpha wrt entropy
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Training Q Function
        predict_target_q1 = self.target_soft_q_net1.forward(next_state, new_next_action)
        predict_target_q2 = self.target_soft_q_net2.forward(next_state, new_next_action)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + self.gamma * target_q_min
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        predict_q1 = self.soft_q_net1.forward(state, new_action)
        predict_q2 = self.soft_q_net2.forward(state, new_action)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def load_model(self):
        self.soft_q_net1 = torch.load('SAC_raw_joint_tactile_less_sandy_navi_soft_q_net1.pkl')
        self.soft_q_net2 = torch.load('SAC_raw_joint_tactile_less_sandy_navi_soft_q_net2.pkl')
        self.target_soft_q_net1 = torch.load('SAC_raw_joint_tactile_less_sandy_navi_target_soft_q_net1.pkl')
        self.target_soft_q_net2 = torch.load('SAC_raw_joint_tactile_less_sandy_navi_target_soft_q_net2.pkl')
        self.policy_net = torch.load('SAC_raw_joint_tactile_less_sandy_navi_policy_net.pkl')

    def save_model(self):
        torch.save(self.soft_q_net1, 'SAC_raw_joint_tactile_less_sandy_navi_soft_q_net1.pkl')
        torch.save(self.soft_q_net2, 'SAC_raw_joint_tactile_less_sandy_navi_soft_q_net2.pkl')
        torch.save(self.target_soft_q_net1, 'SAC_raw_joint_tactile_less_sandy_navi_target_soft_q_net1.pkl')
        torch.save(self.target_soft_q_net2, 'SAC_raw_joint_tactile_less_sandy_navi_target_soft_q_net2.pkl')
        torch.save(self.policy_net, 'SAC_raw_joint_tactile_less_sandy_navi_policy_net.pkl')

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
        self.agent = SAC(state_dim, action_dim)
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
        for param in self.agent.policy_net.parameters():
            mat = param.data.detach().cpu().numpy()
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
            elif mat.ndim == 4:
                if bytes_array == None:
                    bytes_array = self.conv_mat_2_bytes(mat)
                else:
                    bytes_array += self.conv_mat_2_bytes(mat)
        return bytes_array

    def seg_actor_bytes(self, bytes_array):
        temp_list = []
        ind = 0
        while(ind < len(bytes_array)):
            temp_list.append(bytes_array[ind:min(ind+1024, len(bytes_array))])
            ind += 1024
        return temp_list

    def float_2_bytes(self, x):
        # 8 bytes
        return struct.pack('d', x)

    def bytes_2_float(self, x):
        return struct.unpack('d', x)[0]

    def int_2_bytes(self, x):
        # 4 bytes
        return struct.pack(">I", x)

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

    def conv_mat_2_bytes(self, hello):
        bytes_array = None
        for i in range(hello.shape[0]):
            for j in range(hello.shape[1]):
                for k in range(hello.shape[2]):
                    for l in range(hello.shape[3]):
                        if bytes_array == None:
                            bytes_array = self.float_2_bytes(hello[0][0][0][0])
                        else:
                            bytes_array += self.float_2_bytes(hello[i][j][k][l])
        return bytes_array

    def bytes_2_conv_mat(self, bytes_array, dim):
        trans_mat = np.zeros(dim)
        for i in range(dim[0]):
            for j in range(dim[1]):
                for k in range(dim[2]):
                    for l in range(dim[3]):
                        trans_mat[i, j, k, l] = self.bytes_2_float(bytes_array[8*i*dim[1]*dim[2]*dim[3] + 8*j*dim[2]*dim[3] + 8*k*dim[3] + 8*l : 8*i*dim[1]*dim[2]*dim[3] + 8*j*dim[2]*dim[3] + 8*k*dim[3] + 8*l + 8])
        return trans_mat

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim = 3
    action_dim = 1
    batch_size = 64
    max_data_size = 65536  # may change depending on your state/action size
    local_ip = "192.168.1.132"
    local_port = 6666
    replay_buffer = ReplayBuffer(1000000, local_ip, local_port, max_data_size, state_dim, action_dim, batch_size)