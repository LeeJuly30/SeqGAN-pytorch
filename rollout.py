import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import os
import copy

def redistribution( idx, total, min_v):
    idx = (idx + 0.0) / (total + 0.0) * 16.0
    return (np.exp(idx - 8.0) / (1.0 + np.exp(idx - 8.0)))

def rescale( reward, rollout_num=1.0):
    reward = np.array(reward)
    x, y = reward.shape
    ret = np.zeros((x, y))
    for i in range(x):
        l = reward[i]
        rescalar = {}
        for s in l:
            rescalar[s] = s
        idxx = 1
        min_s = 1.0
        max_s = 0.0
        for s in rescalar:
            rescalar[s] = redistribution(idxx, len(l), min_s)
            idxx += 1
        for j in range(y):
            ret[i, j] = rescalar[reward[i, j]]
    return ret

class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate, rescale=False):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.rescale = rescale

    def get_reward(self, data, num, discriminator):
        """
        Args:
            data : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        self.own_model.lstm.flatten_parameters()
        rewards = []
        batch_size = data.size(0)
        seq_len = data.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                sub_sentences = data[:,0:l]
                samples, _ = self.own_model.sample(x=sub_sentences)
                reward = np.exp(discriminator(samples).detach().cpu().data[:,1].numpy())
                if i == 0:
                    rewards.append(reward)
                else:
                    rewards[l-1] += reward
            reward = np.exp(discriminator(data).detach().cpu().data[:,1].numpy())
            if i == 0:
                rewards.append(reward)
            else:
                rewards[seq_len-1] += reward
        if self.rescale:
            rewards = rescale(np.array(rewards), num)
        rewards = torch.from_numpy(np.transpose(np.array(rewards))/(1.0*num)).cuda()
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
