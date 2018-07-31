import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.nn.init as init


class Generator(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, start_token, seq_length, gpu=True, oracle=False):
        super(Generator,self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.gpu = gpu
        self.num_layers = num_layers
        self.start_token = start_token
        self.seq_length = seq_length
        self.device = torch.device("cuda:0" if gpu else "cpu")
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm2out = nn.Linear(hidden_dim, vocab_size-1) # Remove the start token
        self.loss_fn = nn.NLLLoss()
        self.init_params(oracle)

    def init_params(self, oracle):
        if oracle:
            for param in self.parameters():
                param.data.normal_(0, 1)
        else:
            for p in self.parameters():
                p.data.uniform_(-0.05, 0.05)

    def init_hidden(self, batch_size):
        h = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(self.device)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(self.device)
        return h, c

    def processing_data(self, x):
        batch_size, seq_len = x.size()
        inp = torch.zeros(batch_size, seq_len)
        target = x
        inp[:,0] = self.start_token
        inp[:,1:] = target[:,:seq_len-1]
        inp = inp.long().to(self.device)
        target = target.long().to(self.device)
        return inp, target

    def forward(self, x):
        emb = self.emb(x)
        h0, c0 = self.init_hidden(x.size(0))
        output, (h, c) = self.lstm(emb, (h0, c0))
        pred = F.log_softmax(self.lstm2out(output.contiguous().view(-1, self.hidden_dim)),dim=1)
        return pred

    def step(self, x, h, c):
        emb = self.emb(x)
        output, (h, c) = self.lstm(emb, (h, c))
        pred = F.softmax(self.lstm2out(output.view(-1, self.hidden_dim)),dim=1)
        return pred, h, c
    
    def batchNLLLoss(self, x):
        inp, target = self.processing_data(x)
        target = target.contiguous().view(-1)
        pred = self.forward(inp)
        loss = self.loss_fn(pred, target)
        return loss
    
    def sample(self, x=None, num_samples=None, sampleFromZero=True):
        samples = []
        prob = []
        if x is not None:
            sampleFromZero = False
            num_samples = x.size(0)
        h, c = self.init_hidden(num_samples)
        if sampleFromZero:
            x = (torch.ones((num_samples, 1))*self.start_token).long().to(self.device)
            for i in range(self.seq_length):
                state, h, c = self.step(x, h, c)
                x = state.multinomial(1)
                prob_c = state.gather(1, x)
                samples.append(x)
                prob.append(prob_c)
            sentences = torch.cat(samples, 1)
            probs = torch.cat(prob, 1)
            probs = probs.prod(1, keepdim=True)
            return sentences, probs
        else:
            inp, target = self.processing_data(x)
            given_len = inp.size(1)
            lis = inp.chunk(inp.size(1), dim=1)
            groud_truth = target.chunk(target.size(1), dim=1)
            for i in range(given_len):
                state, h, c = self.step(lis[i], h, c)
                samples.append(groud_truth[i])
            x = groud_truth[i]
            for i in range(given_len, self.seq_length):
                state, h, c = self.step(x, h, c)
                x = state.multinomial(1)
                prob_c = state.gather(1, x)
                samples.append(x)
                prob.append(prob_c)
            sentences = torch.cat(samples, 1)
            probs = torch.cat(prob, 1)
            probs = probs.prod(1, keepdim=True)
            return sentences, probs


