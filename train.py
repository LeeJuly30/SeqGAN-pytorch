import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.init as init
import numpy as np
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from dataloader import Gen_Data_loader, Dis_dataloader
import random
import pickle
from multiprocessing import Pool

EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 10 # sequence length
START_TOKEN = 1999
BATCH_SIZE = 64
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88

dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100]
dis_dropout = 0.55
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

TOTAL_BATCH = 200
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000
device = torch.device('cuda:0')

def generate_samples(trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        samples, _ = trainable_model.sample(num_samples=batch_size)
        samples = samples.cpu().data.numpy().tolist()
        generated_samples.extend(samples)

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def target_loss(target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = target_lstm.batchNLLLoss(batch).data.cpu().numpy()
        nll.append(g_loss)

    return np.mean(nll)

def pre_train_epoch(trainable_model, opt, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch().to(device)
        opt.zero_grad()
        g_loss = trainable_model.batchNLLLoss(batch.detach())
        g_loss.backward()
        opt.step()
        supervised_g_losses.append(g_loss.data.cpu().numpy())

    return np.mean(supervised_g_losses)

class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N*C, D), torch Variable 
            target : (N, C), torch Variable
            reward : (N, C), torch Variable
        """
        N = target.size(0)
        C = target.size(1)
        D = prob.size(1)
        one_hot = torch.zeros(N*C, D).float()
        one_hot = one_hot.to(device)
        one_hot.scatter_(1, target.data.contiguous().view(-1, 1), 1)
        reward = reward.contiguous().view(-1, 1)
        loss = one_hot * reward * prob
        loss =  -torch.sum(loss)
        return loss

def calc_bleu(reference, hypothesis, weight):
    return sentence_bleu(reference, hypothesis, weight, smoothing_function=SmoothingFunction().method1)
 
def self_bleu(trainable_model):
    samples, _ = trainable_model.sample(num_samples=200)
    reference = samples.cpu().data.numpy().tolist()
    pool = Pool(os.cpu_count())
    result = list()
    sentence_num = len(reference)
    weight = ((0,0,0.5,0.5))
    for index in range(sentence_num):
        hypothesis = reference[index]
        other = reference[:index] + reference[index+1:]
        result.append(pool.apply_async(calc_bleu, args=(other, hypothesis, weight)))

    score = 0.0
    cnt = 0
    for i in result:
        score += i.get()
        cnt += 1
    pool.close()
    pool.join()
    return score / cnt


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    vocab_size = 2000
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, EMB_DIM, HIDDEN_DIM, 1, START_TOKEN, SEQ_LENGTH).to(device)
    target_lstm = Generator(vocab_size, EMB_DIM, HIDDEN_DIM, 1, START_TOKEN, SEQ_LENGTH, oracle=True).to(device)
    discriminator = Discriminator(vocab_size, dis_embedding_dim, dis_filter_sizes, dis_num_filters, dis_dropout).to(device)

    generate_samples(target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)

    pre_gen_opt = torch.optim.Adam(generator.parameters(), 1e-2)
    adv_gen_opt = torch.optim.Adam(generator.parameters(), 1e-2)
    dis_opt = torch.optim.Adam(discriminator.parameters(), 1e-4)
    dis_criterion = nn.NLLLoss()

    log = open('save/experiment-log.txt', 'w')
    print('Start pre-training...')
    log.write('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        loss = pre_train_epoch(generator, pre_gen_opt, gen_data_loader)
        if (epoch+1) % 5 == 0:
            generate_samples(generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(target_lstm, likelihood_data_loader)
            print('pre-train epoch ', epoch+1, '\tnll:\t', test_loss)
            buffer = 'epoch:\t'+ str(epoch+1) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)
    print('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    for e in range(50):
        generate_samples(generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        d_total_loss = []
        for _ in range(3):
            dis_data_loader.reset_pointer()
            total_loss = []
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                dis_output = discriminator(x_batch.detach())
                d_loss = dis_criterion(dis_output, y_batch.detach())
                dis_opt.zero_grad()
                d_loss.backward()
                dis_opt.step()
                total_loss.append(d_loss.data.cpu().numpy())
            d_total_loss.append(np.mean(total_loss))
        if (e+1) % 5 == 0:
            buffer = 'Epoch [{}], discriminator loss [{:.4f}]\n'.format(e+1, np.mean(d_total_loss))
            print(buffer)
            log.write(buffer)

    rollout = Rollout(generator, 0.8)
    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    gan_loss = GANLoss()
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        discriminator.eval()
        for it in range(1):
            samples, _ = generator.sample(num_samples=BATCH_SIZE)
            rewards = rollout.get_reward(samples, 16, discriminator)
            prob = generator(samples.detach())
            adv_loss = gan_loss(prob, samples.detach(), rewards.detach())
            adv_gen_opt.zero_grad()
            adv_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            adv_gen_opt.step()

        # Test
        if (total_batch+1) % 5 == 0:
            generate_samples(generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(target_lstm, likelihood_data_loader)
            self_bleu_score =  self_bleu(generator)
            buffer = 'epoch:\t' + str(total_batch+1) + '\tnll:\t' + str(test_loss) + '\tSelf Bleu:\t' + str(self_bleu_score) + '\n'
            print(buffer)
            log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        discriminator.train()
        for _ in range(5):
            generate_samples(generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            d_total_loss = []
            for _ in range(3):
                dis_data_loader.reset_pointer()
                total_loss = []
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    dis_output = discriminator(x_batch.detach())
                    d_loss = dis_criterion(dis_output, y_batch.detach())
                    dis_opt.zero_grad()
                    d_loss.backward()
                    dis_opt.step()
                    total_loss.append(d_loss.data.cpu().numpy())
                d_total_loss.append(np.mean(total_loss))
            if (total_batch+1) % 5 == 0:
                buffer = 'Epoch [{}], discriminator loss [{:.4f}]\n'.format(total_batch+1, np.mean(d_total_loss))
                print(buffer)
                log.write(buffer)
    log.close()

if __name__ == '__main__':
    main()
