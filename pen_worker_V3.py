import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import socket
import threading
import struct
import time

class BufferAgent:
    def __init__(self, buffer_ip, buffer_port, local_ip, local_port, state_dim, action_dim, batch_size, max_data_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = SAC(state_dim, action_dim)
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
        self.client_socket.send(self.int_2_bytes(6) + self.int_2_bytes(0))
        # print('ready to catch', self.max_data_size, 'data')
        # print('expected data', self.actor_len())
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
        # print('received data', len(temp_data))

        ind = 0
        for param in self.agent.policy_net.parameters():
            mat = param.data.detach().numpy()
            if len(mat.shape) == 1:
                new_mat = self.bytes_2_array(temp_data[ind:ind + 8 * mat.shape[0]], mat.shape[0])
                param.data = nn.parameter.Parameter(torch.from_numpy(new_mat).float())
                ind += 8 * mat.shape[0]
            elif len(mat.shape) == 2:
                # print('size 2')
                new_mat = self.bytes_2_mat(temp_data[ind:ind + 8 * mat.shape[0] * mat.shape[1]],
                                           (mat.shape[0], mat.shape[1]))
                param.data = nn.parameter.Parameter(torch.from_numpy(new_mat).float())
                ind += 8 * mat.shape[0] * mat.shape[1]
            elif len(mat.shape) == 4:
                new_mat = self.bytes_2_conv_mat(temp_data[ind:ind + 8 * mat.shape[0] * mat.shape[1] * mat.shape[2] * mat.shape[3]],
                                           (mat.shape[0], mat.shape[1], mat.shape[2], mat.shape[3]))
                param.data = nn.parameter.Parameter(torch.from_numpy(new_mat).float())
                ind += 8 * mat.shape[0] * mat.shape[1] * mat.shape[2] * mat.shape[3]
        # print('total layers', i)
        print('synchronize actor')

    def actor_len(self):
        a = 0
        for param in self.agent.policy_net.parameters():
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

    def bytes_2_conv_mat(self, bytes_array, dim):
        trans_mat = np.zeros(dim)
        for i in range(dim[0]):
            for j in range(dim[1]):
                for k in range(dim[2]):
                    for l in range(dim[3]):
                        trans_mat[i, j, k, l] = self.bytes_2_float(bytes_array[8*i*dim[1]*dim[2]*dim[3] + 8*j*dim[2]*dim[3] + 8*k*dim[3] + 8*l : 8*i*dim[1]*dim[2]*dim[3] + 8*j*dim[2]*dim[3] + 8*k*dim[3] + 8*l + 8])
        return trans_mat

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

        for param in self.policy_net.parameters():
            mat = param.data.detach().cpu().numpy()
            print(mat.shape)

    def get_action(self, state):
        a = self.policy_net.get_action(state)
        return a

    def train(self, batch):
        state, action, reward, next_state, done = batch

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(-1)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1)

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

if __name__ == '__main__':
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    state_dim = 3
    action_dim = 1
    max_epi_iter = 200
    max_MC_iter = 200
    batch_size = 64
    local_ip = "192.168.1.132"
    local_port = 6669
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
            action1 = buffer_agent.agent.get_action(state)
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
        if epi % 10 == 0:
            buffer_agent.sync_actor()
        # np.save('data/remote_con_train_sync_each_epi_sleep_2_1.npy', np.array(train_curve))