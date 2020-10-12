import os
import os.path as osp
import argparse
import numpy as np 
import json
import time
 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models                                                                     

from coco_loader import coco_loader
from convcap import convcap
from vggfeats import Vgg16Feats
from tqdm import tqdm 
from test import test 
from seq_model import *
from rl_utils5_perception import *
import misc.utils as utils
rl_crit = utils.RewardCriterion()



def repeat_img_per_cap(imgsfeats, imgsfc7, ncap_per_img):
  """Repeat image features ncap_per_img times"""

  batchsize, featdim, feat_h, feat_w = imgsfeats.size()
  batchsize_cap = batchsize*ncap_per_img
  imgsfeats = imgsfeats.unsqueeze(1).expand(\
    batchsize, ncap_per_img, featdim, feat_h, feat_w)
  imgsfeats = imgsfeats.contiguous().view(\
    batchsize_cap, featdim, feat_h, feat_w)
  
  batchsize, featdim = imgsfc7.size()
  batchsize_cap = batchsize*ncap_per_img
  imgsfc7 = imgsfc7.unsqueeze(1).expand(\
    batchsize, ncap_per_img, featdim)
  imgsfc7 = imgsfc7.contiguous().view(\
    batchsize_cap, featdim)

  return imgsfeats, imgsfc7

def train(args):
  """Trains model for args.nepochs (default = 30)"""
 
  t_start = time.time()
  train_data = coco_loader(args.coco_root, split='train', ncap_per_img=args.ncap_per_img)
  print('[DEBUG] Loading train data ... %f secs' % (time.time() - t_start))

  train_data_loader = DataLoader(dataset=train_data, num_workers=args.nthreads,\
    batch_size=args.batchsize, shuffle=True, drop_last=True)

  lang_model = Seq2Seq(train_data.numwords)
  lang_model =lang_model.cuda()
  lang_model.load_state_dict(torch.load('log_model/bestmodel.pth')['lang_state_dict'])
  lang_model.train() 
 #Load pre-trained imgcnn
  model_imgcnn = Vgg16Feats()  
  model_imgcnn.cuda() 
  model_imgcnn.train(True) 
  model_imgcnn.load_state_dict(torch.load('log_reg/bestmodel.pth')['img_state_dict'])
  #Convcap model
  model_convcap = convcap(train_data.numwords, args.num_layers, is_attention=args.attention)
  model_convcap.cuda()
  model_convcap.load_state_dict(torch.load('log_reg/bestmodel.pth')['state_dict'])
  model_convcap.train(True)

  optimizer = optim.RMSprop(model_convcap.parameters(), lr=args.learning_rate)
  scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=.1)
  img_optimizer = None

  batchsize = args.batchsize
  ncap_per_img = args.ncap_per_img
  batchsize_cap = batchsize*ncap_per_img
  max_tokens = train_data.max_tokens
  nbatches = np.int_(np.floor((len(train_data.ids)*1.)/batchsize)) 
  bestscore = .0

  for epoch in range(args.epochs):
    loss_train = 0.
    
    if(epoch == args.finetune_after):
      img_optimizer = optim.RMSprop(model_imgcnn.parameters(), lr=1e-5)
      img_scheduler = lr_scheduler.StepLR(img_optimizer, step_size=args.lr_step_size, gamma=.1)

    scheduler.step()    
    if(img_optimizer):
      img_scheduler.step()
    it = 0
    #One epoch of train
    for batch_idx, (imgs, captions, wordclass, mask, _) in \
      tqdm(enumerate(train_data_loader), total=nbatches):
      it = it + 1
      imgs = imgs.view(batchsize, 3, 224, 224)
      wordclass = wordclass.view(batchsize_cap, max_tokens).cuda()
      mask = mask.view(batchsize_cap, max_tokens)

      captions = utils.decode_sequence(train_data.wordlist, wordclass, None)
      captions_all = []
      for index, caption in enumerate(captions):
          captions_all.append(caption)




      imgs_v = Variable(imgs).cuda()
      wordclass_v = Variable(wordclass).cuda()

      optimizer.zero_grad()
      if(img_optimizer):
        img_optimizer.zero_grad() 

      imgsfeats, imgsfc7 = model_imgcnn(imgs_v)
      imgsfeats, imgsfc7 = repeat_img_per_cap(imgsfeats, imgsfc7, ncap_per_img)
      _, _, feat_h, feat_w = imgsfeats.size()

      if(args.attention == True):
        wordact, attn = model_convcap(imgsfeats, imgsfc7, wordclass_v)
        attn = attn.view(batchsize_cap, max_tokens, feat_h, feat_w)
      else:
        wordact, _ = model_convcap(imgsfeats, imgsfc7, wordclass_v)

      wordact = wordact[:,:,:-1]
      wordclass_v = wordclass_v[:,1:]
      mask = mask[:,1:].contiguous()

      wordact_t = wordact.permute(0, 2, 1).contiguous().view(\
        batchsize_cap*(max_tokens-1), -1)
      wordclass_t = wordclass_v.contiguous().view(\
        batchsize_cap*(max_tokens-1), 1)
      
      maskids = torch.nonzero(mask.view(-1)).numpy().reshape(-1)

      if(args.attention == True):
        #Cross-entropy loss and attention loss of Show, Attend and Tell
        loss_xe = F.cross_entropy(wordact_t[maskids, ...], \
          wordclass_t[maskids, ...].contiguous().view(maskids.shape[0])) \
          + (torch.sum(torch.pow(1. - torch.sum(attn, 1), 2)))\
          /(batchsize_cap*feat_h*feat_w)
      else:
        loss_xe = F.cross_entropy(wordact_t[maskids, ...], \
          wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]))
     
     
      wordact = lang_model(wordclass_v.transpose(1,0), wordclass_v.transpose(1,0), imgs)
      wordact = wordact.transpose(1,0)[:,:-1,:]
      wordclass_v = wordclass_v[:,1:]

      wordact_t = wordact.contiguous().view(\
        batchsize_cap*wordact.size(1), -1)

      wordclass_t = wordclass_v.contiguous().view(\
        batchsize_cap*wordclass_v.size(1), 1)

      loss_xe_lang = F.cross_entropy(wordact_t[...], \
          wordclass_t[...].contiguous().view(-1))
     
      with torch.no_grad():
          outcap, sampled_ids, sample_logprobs, x_all_langauge, outputs = lang_model.sample(wordclass.transpose(1,0), wordclass.transpose(1,0), imgsfeats.transpose(1,0), train_data.wordlist)
      
      logprobs_input,_ = model_convcap(imgsfeats, imgsfc7, sampled_ids.long().cuda())
      log_probs = F.log_softmax(logprobs_input.transpose(2,1)[:,:-1,:], -1)

      sample_logprobs_true = log_probs.gather(2, sampled_ids[:, 1:].cuda().long().unsqueeze(2))
      with torch.no_grad():
          reward = get_self_critical_reward(batchsize_cap, lang_model, wordclass.transpose(1,0), imgsfeats.transpose(1,0), outcap, captions_all, train_data.wordlist, 16)

      
      loss_rl1 = rl_crit(torch.exp(sample_logprobs_true.squeeze())/torch.exp(sample_logprobs[:,1:]).cuda().detach(), sampled_ids[:, 1:].cpu(), torch.from_numpy(reward).float().cuda())
      #loss_rl2 = rl_crit(sample_logprobs[:,1:].cuda(), sampled_ids[:, 1:].cpu(), torch.from_numpy(reward).float().cuda())

      loss = 0.0*loss_xe + loss_rl1# + loss_xe_lang + loss_rl2
       
      if it % 500 ==0:
          modelfn = osp.join(args.model_dir, 'model.pth')
          scores = test(args, 'val', model_convcap=model_convcap, model_imgcnn=model_imgcnn)
          score = scores[0][args.score_select]
          if(score > bestscore):
              bestscore = score
              print('[DEBUG] Saving model at epoch %d with %s score of %f'\
                 % (epoch, args.score_select, score))
              bestmodelfn = osp.join(args.model_dir, 'bestmodel.pth')
              os.system('cp %s %s' % (modelfn, bestmodelfn))

          torch.save({
           'epoch': epoch,
           'state_dict': model_convcap.state_dict(),
           'img_state_dict': model_imgcnn.state_dict(),
           'optimizer' : optimizer.state_dict(),
           'lang_state_dict': lang_model.state_dict()
                 }, modelfn)


      loss_train = loss_train + loss

      loss.backward()

      optimizer.step()
      if(img_optimizer):
        img_optimizer.step()

    loss_train = (loss_train*1.)/(batch_idx)
    print('[DEBUG] Training epoch %d has loss %f' % (epoch, loss_train))

    modelfn = osp.join(args.model_dir, 'model.pth')

    if(img_optimizer):
      img_optimizer_dict = img_optimizer.state_dict()
    else:
      img_optimizer_dict = None

    torch.save({
        'epoch': epoch,
        'state_dict': model_convcap.state_dict(),
        'img_state_dict': model_imgcnn.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'lang_state_dict': lang_model.state_dict()
      }, modelfn)

    #Run on validation and obtain score
    scores = test(args, 'val', model_convcap=model_convcap, model_imgcnn=model_imgcnn) 
    score = scores[0][args.score_select]

    if(score > bestscore):
      bestscore = score
      print('[DEBUG] Saving model at epoch %d with %s score of %f'\
        % (epoch, args.score_select, score))
      bestmodelfn = osp.join(args.model_dir, 'bestmodel.pth')
      os.system('cp %s %s' % (modelfn, bestmodelfn))
