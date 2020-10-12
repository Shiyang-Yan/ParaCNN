#= This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import math
from .CaptionModel import CaptionModel
import copy

import pdb

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def Conv1d(in_channels, out_channels, kernel_size, padding, dropout=0):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m

def Linear(in_features, out_features, dropout=0.):
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

class AttentionLayer_context(nn.Module):
  def __init__(self):
    super(AttentionLayer_context, self).__init__()
    self.tanh = nn.Tanh()
    self.v = nn.Linear(512, 1)
    self.softmax = nn.Softmax(-1)
  def forward(self, x, imgsfeats):
    x = x.unsqueeze(2) #10, 512
    y = imgsfeats.transpose(2, 1) # 10, 512, 41
    att = self.tanh(x + y)
    att = att.permute(0,2,1)
    e = self.v(att).squeeze(2)
    alpha = self.softmax(e) #10, 41
    context = (y * alpha.unsqueeze(1)).sum(2) #10, 512
    return context



class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.tanh = nn.Tanh()
        self.v = nn.Linear(512, 1)
        self.softmax = nn.Softmax(-1)

    def forward(self, x, wordemb, imgsfeats):
         x = x.unsqueeze(3)  # 10, 30, 512
         wordemb = wordemb.unsqueeze(3)
         y = imgsfeats.transpose(2, 1)  # 10, 512, 41
         y = y.unsqueeze(1).expand(y.size(0),30,512,y.size(2)) # 10, 30, 512, 41
         att = self.tanh(x + y + wordemb)
         att = att.permute(0, 1, 3, 2)
         e = self.v(att).squeeze(3)
         alpha = self.softmax(e)  # 10, 30, 41
         context = (y * alpha.unsqueeze(2)).sum(3)  # 10, 512
         return context, alpha

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        #self.f_linear = nn.Linear(d_model, 256)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores/150, dim = -1)
        if dropout is not None:
           p_attn = dropout(p_attn)
        out = torch.matmul(p_attn, value).mean(3)
        return out, p_attn


    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches,6,  -1, self.h, self.d_k).transpose(2,3)
             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(2,3).contiguous() \
             .view(nbatches,6, -1, self.h * self.d_k)
        #out = F.relu(self.f_linear(F.relu(out)))
        return self.linears[-1](x), self.attn


def multinomial(probs=None, logits=None, temperature=1, num_samples=1,
                     min_prob=1e-20, max_logit=1e+20,
                     min_temperature=1e-20, max_temperature=1e+20):
    if probs is not None:
        logits = torch.log(probs)
    probs = torch.exp(logits).detach()
    with torch.no_grad():
      sample = torch.multinomial(probs, num_samples)
    return sample, logits

class convcap(nn.Module):
  
  def __init__(self, opt):
    super(convcap, self).__init__()
    num_wordclass = opt.vocab_size + 1
    num_layers = opt.num_layers
    self.att = MultiHeadedAttention(8, 512)
    is_attention = True
    nfeats = 1024
    dropout = .1
    self.fc_imgfeats = Linear(2048, nfeats//2)
    self.nimgfeats = 2048
    self.is_attention = is_attention
    self.nfeats = nfeats
    self.dropout = dropout 
    self.attention_context = AttentionLayer_context()
    self.emb_0 = Embedding(num_wordclass, nfeats//2, padding_idx=0)
    self.emb_1 = Linear(nfeats//2, nfeats//2, dropout=dropout)
    self.topic_proj = Linear(self.nfeats, self.nfeats//2)
    self.imgproj = Linear(self.nfeats//2, self.nfeats//2, dropout=dropout)
    self.resproj = Linear(nfeats, self.nfeats//2, dropout=dropout)
    self.resproj_para = Linear(nfeats//2, self.nfeats//2 ,dropout=dropout)
    n_in = 2*self.nfeats
    n_out = self.nfeats
    self.n_layers = 3
    self.convs = nn.ModuleList()
    self.attention = nn.ModuleList()
    self.kernel_size = 5
    self.pad = self.kernel_size - 1
    for i in range(self.n_layers):
      self.convs.append(Conv1d(n_in//2, 2*n_out//2, self.kernel_size, self.pad, dropout))
      if(self.is_attention and i%2==0):
        self.attention.append(AttentionLayer())
      n_in = n_out
    self.kernel_topic = 5
    self.pad_topic = self.kernel_topic -1
    self.convs_topic = nn.ModuleList()
    for i in range(4):
      self.convs_topic.append(Conv1d(n_in//2, 2*n_out//2, self.kernel_topic, self.pad_topic))
      n_in = n_out

    self.classifier_0 = Linear(self.nfeats//2, (nfeats // 2))
    #self.bn1 = nn.BatchNorm1d(num_features = (nfeats // 2))
    self.classifier_1 = Linear((nfeats // 2), num_wordclass, dropout=dropout)

  def conv_topic(self, x_all):
    for i, conv_topic in enumerate(self.convs_topic):
      
        if(i==0):
          x_all = x_all.transpose(2,1)
          residual = self.resproj_para(x_all)
          residual = residual.transpose(2,1)
          x_all = x_all.transpose(2,1)
        else:
          residual = x_all

        x_all = conv_topic(x_all)
        x_all = x_all[:,:,:-self.pad_topic]

        x_all = F.glu(x_all, dim=1)
        x_all = (x_all+residual)*math.sqrt(.5)
    return x_all

  def conv_cap(self, x, wordemb, imgsfeats):
    for i, conv in enumerate(self.convs):
      
        if(i == 0):
          x = x.transpose(2, 1)
          residual = self.resproj(x)
          residual = residual.transpose(2, 1)
          x = x.transpose(2, 1)
        else:
          residual = x

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = conv(x)
        x = x[:,:,:-self.pad]

        x = F.glu(x, dim=1)

        if(self.is_attention and i%2 == 0):
          attn = self.attention[int(i/2)]
          x = x.transpose(2, 1)
          x, alpha = attn(x, wordemb, imgsfeats)
          x = F.relu(x.transpose(2, 1))
    
        x = (x+residual)*math.sqrt(.5)
    return x, alpha
  def sampling(self, imgsfeats, wordclass):
      outcap = np.empty((batchsize_cap, 0)).tolist()
      imgsfeats_used = F.relu(self.fc_imgfeats(imgsfeats))

      wordclass = wordclass
      imgsfc7, _ = imgsfeats_used.max(1)
      imgsfc7 = imgsfc7
      batchsize = imgsfc7.size(0)

      output = torch.zeros((batchsize, num, 8668, length))
      context = torch.zeros((batchsize, 512))
      para_level_visual = imgsfc7.unsqueeze(2).expand(imgsfc7.size(0), self.nfeats // 2, num)
      x_all = torch.zeros((batchsize, length, num, 512))
      x_all_out = torch.zeros((batchsize, length, num, 512))
      for index in range(num):
          if index > 0:
              x_all[:, :, index, :] = x2

          x_all2 = x_all.transpose(2, 1).contiguous().cuda()
          # x_all_para = torch.mean(x_all, 1).cuda().transpose(2,1)
          x_all_para = self.att(x_all2, x_all2, x_all2)
          x_all_para = x_all_para.view(batchsize, 6, 512).transpose(2, 1).cuda()
          x_tobe_conv = torch.cat([x_all_para, para_level_visual], 1)

          x_tobe_conv = F.relu(self.topic_proj(x_tobe_conv.transpose(2, 1)))
          topic_conv = self.conv_topic(x_tobe_conv.transpose(2, 1))

          topic = F.relu(topic_conv[:, :, index])

          wordemb = self.emb_0(wordclass[:, index, :].squeeze())
          wordemb = self.emb_1(wordemb)
          if len(wordemb.size()) > 2:
              x = wordemb.transpose(2, 1)
              batchsize, wordembdim, maxtokens = x.size()
          else:
              x = wordemb.transpose(1, 0).unsqueeze(0)
              batchsize, wordembdim, maxtokens = x.size()

          y = F.relu(self.imgproj(topic))

          y = y.unsqueeze(2).expand(batchsize, self.nfeats // 2, maxtokens)
          x = torch.cat([x, y], 1)

          x = self.conv_cap(x, wordemb, imgsfeats_used)
          x = x.transpose(2, 1)

          x = self.classifier_0(x)

          x = x.transpose(2, 1)

          x1 = F.relu(x)

          x1 = x1.transpose(2, 1)

          x2 = F.dropout(x1, p=self.dropout, training=self.training)
          x_all_out[:, :, index, :] = x2

          x3 = self.classifier_1(x2)
          x3 = x3.transpose(2, 1)
          output[:, index, :] = x3
      if len(output.size()) > 2:
          output = output.transpose(2, 1).contiguous()
      else:
          output = output.transpose(1, 0).unsqueeze(0)
      output = output.view((batchsize, 8668, -1))
      output = output.transpose(2,1).contiguous().view(-1, 8668)
      wordact_t_prob = F.softmax(output)
      sampled_distribution, logits = multinomial(probs=wordact_t_prob)
      sampleLogprobs = logits.gather(1, sampled_distribution)
      sample_probs = sampleLogprobs.view(batchsize, 180)
      sample_dis = sampled_distribution.view(batchsize, 180)
      with torch.no_grad():
        for j in range(180):
          for k in range(batchsize):
              index = str(int(sample_probs[k].data))
              if int(sample_dis[k].data) > 0 and int(sample_dis[k].data) != 8667 and int(
                      sample_dis[k].data) != 8666:
                  word = ix_to_word[index]
                  outcap[k].append(word)
              if (j < max_tokens - 1):
                  wordclass_feed[k, j + 1] = sample_dis[k]
      return outcap, sampled_distribution, sample_probs

  def forward(self, non_used, imgsfeats, wordclass, length, num):

    imgsfeats_used = F.relu(self.fc_imgfeats(imgsfeats)) 

    wordclass = wordclass
    imgsfc7,_ = imgsfeats_used.max(1)
    imgsfc7 = imgsfc7
    batchsize = imgsfc7.size(0)

    output = torch.zeros((batchsize, num, 8668, length))
    context = torch.zeros((batchsize, 512))
    para_level_visual = imgsfc7.unsqueeze(2).expand(imgsfc7.size(0), self.nfeats//2, num)
    x_all = torch.zeros((batchsize, length, num , 512))
    x_all_out = torch.zeros((batchsize, length, num, 512))
    for index in range(num):
      if index >0:

        x_all[:, :, index, :] = x2
      
      x_all2 = x_all.transpose(2,1).contiguous().cuda()
      #x_all_para = torch.mean(x_all, 1).cuda().transpose(2,1)
      x_all_para, self_alpha = self.att(x_all2, x_all2, x_all2)
      x_all_para = x_all_para.view(batchsize, 6, 512).transpose(2,1).cuda()
      x_tobe_conv = torch.cat([x_all_para,para_level_visual], 1)
   
      x_tobe_conv = F.relu(self.topic_proj(x_tobe_conv.transpose(2,1)))
      topic_conv = self.conv_topic(x_tobe_conv.transpose(2,1))

      topic = F.relu(topic_conv[:,:,index])

      wordemb = self.emb_0(wordclass[:,index,:])
      wordemb = self.emb_1(wordemb)
      if len(wordemb.size())>2:
        x = wordemb.transpose(2, 1)  
        batchsize, wordembdim, maxtokens = x.size()
      else:
        x = wordemb.transpose(1, 0).unsqueeze(0)
        batchsize, wordembdim, maxtokens = x.size()

      y = F.relu(self.imgproj(topic))

      y = y.unsqueeze(2).expand(batchsize, self.nfeats//2, maxtokens)
      x = torch.cat([x, y], 1)

      x, vis_alpha = self.conv_cap(x, wordemb, imgsfeats_used)
      #print (vis_alpha.size())


      x = x.transpose(2, 1)
     
      x = self.classifier_0(x)  

      x = x.transpose(2, 1)

      x1 = F.relu(x)
      
      x1 = x1.transpose(2,1)

      x2 = F.dropout(x1, p=self.dropout, training=self.training)
      x_all_out[:,:,index,:] = x2

      x3 = self.classifier_1(x2)
      x3 = x3.transpose(2, 1)
      output[:, index,:] = x3
    if len(output.size())>2:
        output = output.transpose(2,1).contiguous()
    else:
        output = output.transpose(1,0).unsqueeze(0)
    output = output.view((batchsize, 8668, -1))
    x_all_out = x_all_out.permute(0,3,2,1).contiguous()
    x_all_out = x_all_out.view((batchsize, 512, -1)).contiguous()
    return output, x_all_out, self_alpha

