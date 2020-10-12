import os
import os.path as osp
import argparse
import numpy as np 
import json
import time
from nltk import ngrams

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm 

from collections import OrderedDict
from sample import sample_greedy_orig
from random import randint

import numpy as np
import torch
import sys
sys.path.append("coco-caption")
from pycocoevalcap.cider.cider import Cider
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu
import sys
CiderD_scorer = None
Bleu_scorer = None
Cider_scorer = Cider()

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()
def init_scorer(cached_tokens = 'coco-train-idxs'):
    global Cider_scorer
    Cider_scorer = Cider_scorer or Cider()
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)



def unique_score(result, batchsize):
    n = 3

    count_all = []
    num_all = []
    for i in range(2*batchsize):
      trigram = [x for x in ngrams(result[i][0].split(),n)]
      splitting = set(trigram)
      num = len(trigram)
      num_all.append(num)
      count = len(splitting)
      count_all.append(count)
    
    unique_rate = np.divide(np.array(count_all).astype(float), np.array(num_all).astype(float)).astype(float)
    return unique_rate


def get_self_critical_reward(batchsize, model, labels, img,gen_result, captions, data, max_tokens):
   # batchsize = gen_result.size(0)# batch_size = sample_size * seq_per_img
    # get greedy decoding baseline
    seq_per_img = 1
    with torch.no_grad():
      outcaps = model.sample_greedy(labels, labels, img, data.get_vocab())
      res = OrderedDict()
      gen_result = np.array(gen_result)
      greedy_res = np.array(outcaps)
      for i in range(batchsize):
          res[i] = [array_to_str(gen_result[i])]
      for i in range(batchsize):
          res[batchsize + i] = [array_to_str(greedy_res[i])]
      res__ = {i: res[i] for i in range(2 * batchsize)}
      #unique_rate = unique_score(res__, batchsize)
      gts = OrderedDict()
      print (res__[0], res__[batchsize])
      for i in range(batchsize):
         gts[i] = [captions[i]]
      for i in range(batchsize):
         gts[batchsize + i] = [captions[i]]
      #print (res__[batchsize], gts[0])
      init_scorer()
      cider, cider_scores = Cider_scorer.compute_score(gts, res__)
      #bleu_score, bleu_scores = Bleu_scorer.compute_score(gts, res__)
      #bleu_scores = np.array(bleu_scores[3])

      scores = cider_scores# + bleu_scores
      scores_out = scores[:batchsize] - scores[batchsize:]
      scores_out = scores_out.reshape(gen_result.shape[0])
      rewards = np.repeat(scores_out[:, np.newaxis], 179, 1)
      return rewards, scores[:batchsize], scores[batchsize:]

##################################

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

