import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

def sample_greedy_orig(batchsize_cap, max_tokens, model, ix_to_word, captions):
    outcap = np.empty((batchsize_cap, 0)).tolist()
    max_tokens = 180
    wordclass_feed = np.zeros((batchsize_cap, max_tokens), dtype='int64')
    with torch.no_grad():
      wordclass_feed[:, 0] = 8667
      for j in range(max_tokens - 1):
        wordact, _ = model.sample(captions, torch.tensor(wordclass_feed).long().cuda(), j)
        wordact_t = wordact.contiguous().view(batchsize_cap, -1)
        wordprobs = F.softmax(wordact_t, -1)
        sampled_ids = torch.argmax(wordprobs, -1)
        for k in range(batchsize_cap):
            index = sampled_ids[k]
            if index > 0 and index != 8667 and index != 8666:
                word = ix_to_word[str(int(index))]
                outcap[k].append(word)
            if (j < max_tokens - 1):
                wordclass_feed[k, j + 1] = sampled_ids[k]
      return outcap
def multinomial(probs=None, temperature=1, num_samples=1,
                     min_prob=1e-20, max_logit=1e+20,
                     min_temperature=1e-20, max_temperature=1e+20):
    if probs is not None:
        #probs = probs.clamp(min=min_prob)
        logits = probs
    #logits = logits.clamp(max=max_logit)
    #temperature = np.clip(temperature, min_temperature, max_temperature)
    #logits = (logits - logits.max()) / temperature
    probs = torch.exp(logits).detach()
    with torch.no_grad():
      sample = torch.multinomial(probs, num_samples)
    return sample


def samplenet(model,  batchsize_cap, max_tokens, ix_to_word, imgsfeats):
        states = torch.zeros(batchsize_cap, 512)
        if 1:
          outcap = np.empty((batchsize_cap, 0)).tolist()
          wordprobs_final = torch.zeros((batchsize_cap, max_tokens-1))
          target = torch.zeros((batchsize_cap, max_tokens-1, 1)).float()
          wordclass_feed = np.zeros((batchsize_cap, max_tokens), dtype='int64')
          wordclass_feed[:,0] = 8667
          x_all = torch.zeros((batchsize_cap, 512, max_tokens))
        for j in range(max_tokens-1):
            print (j)
            if j%30 ==0:
              wordclass_feed[:, j] = 8667
            wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()
            states, wordact, x_current = model.sample(wordclass, states)
            x_all[:,:, j] = x_current.squeeze()
            wordact_m = wordact
            #should be wordact[:,:,1:] for the last computation
            if 1:
              wordact_t = wordact_m.contiguous().view(batchsize_cap, -1)
              wordact_t_prob = F.log_softmax(wordact_t) #positive
              wordact_t_prob_s = wordact_t_prob.view(batchsize_cap, -1)
              probs_s = wordact_t_prob_s #postive
              sampled_distribution = multinomial(probs = probs_s) #index
              target[:, j, :] = sampled_distribution
              sampleLogprobs = probs_s.gather(1, sampled_distribution) # gather the logprobs at sampled positions
              wordprobs_final[:, j] = sampleLogprobs.squeeze()
            with torch.no_grad():
             for k in range(batchsize_cap):   
                index = str(int(sampled_distribution[k].data))
                if int(sampled_distribution[k].data)> 0 and int(sampled_distribution[k].data) != 8667 and int(sampled_distribution[k].data) != 8666:
                   word = ix_to_word[index]
                   outcap[k].append(word)
                if(j < max_tokens-1):
                   wordclass_feed[k, j+1] = sampled_distribution[k]
        return  target.squeeze(), outcap,  wordprobs_final, x_all
