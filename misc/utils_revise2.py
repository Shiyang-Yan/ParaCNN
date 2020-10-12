from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq, prob):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        sent = 0
        for j in range(D):
            if j>0 and j%30 ==0:
              sent = sent + 1
            if 1:
              ix = seq[i,j]
              #ind = prob[i, sent]
              if 1:
                if ix.item() > 0 :
                    if j >= 1:
                        txt = txt + ' '
                    txt = txt + ix_to_word[str(ix.item())]
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()






class RewardCriterion2(nn.Module):
    def __init__(self):
        super(RewardCriterion2, self).__init__()


    def forward(self, seqLogprobs, reward, batchsize_cap, target):
      print (seqLogprobs.size())
      seqLogprobs = seqLogprobs.contiguous().view(batchsize_cap*(180-1), -1).cuda()
      one_hot = torch.zeros(seqLogprobs.size()).float()
      one_hot.scatter_(1, target.view((-1,1)).long(), 1)
      one_hot = one_hot.type(torch.ByteTensor)
      one_hot = Variable(one_hot)
      if seqLogprobs.is_cuda:
          one_hot = one_hot.cuda()
      loss = torch.masked_select(seqLogprobs, one_hot)
      loss = loss * torch.Tensor(reward).cuda().float().contiguous().view(-1)
      loss =  -torch.sum(loss)
      
      return loss







class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
    
