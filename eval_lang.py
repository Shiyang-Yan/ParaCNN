from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils_trigram
import argparse
import misc.utils as utils
import torch
import opts
from seq_model import *
from rl_utils import *
from sample import *

opt = opts.parse_opt()


opt.batch_size =13
opt.input_att_dir =  'data/parabu_att'
opt.input_fc_dir = 'data/parabu_fc'
opt.input_json = 'data/paratalk.json'
opt.input_label_h5 = 'data/paratalk_label.h5'
opt.language_eval =1
opt.learning_rate = 4e-4
opt.learning_rate_decay_start =0
opt.scheduled_sampling_start =0
opt.max_epochs= 80
opt.save_checkpoint_every =10000
opt.checkpoint_path= 'log_xe/'
opt.print_freq =100
opt.model = 'log_cvpr/langmodel80000.pth'
opt.id = 'xe'



# Load infos
loader = DataLoader(opt)
opt.vocab_size = loader.vocab_size
opt.seq_length = loader.seq_length
opt.input_fc_dir = opt.input_fc_dir
opt.input_att_dir = opt.input_att_dir
opt.input_label_h5 = opt.input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = opt.input_json
if opt.batch_size == 0:
    opt.batch_size = opt.batch_size
if len(opt.id) == 0:
    opt.id = opt.id
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval", "block_trigrams", "alpha"]


lang_model = auto_encoder().cuda()
lang_model.load_state_dict(torch.load(opt.model))
lang_model.eval()

while True:

    data = loader.get_batch('test')
    n = loader.batch_size

    tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['dist'], data['masks'], data['att_masks']]
    tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
    fc_feats, att_feats, labels, dist_label, masks, _ = tmp
    # forward the model to also get generated samples for each image
    num = 6
    length = 30
    max_tokens = num * length
    with torch.no_grad():
        outcaps = lang_model.sample(labels.view(fc_feats.size(0), -1), loader.get_vocab())
        print (outcaps)    # Print beam search
    # print (prob)

