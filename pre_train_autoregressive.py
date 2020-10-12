from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torch.nn.functional as F
import time
import os
from six.moves import cPickle
import math

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
from seq_model import *

def _calcualte_discriminator_loss(tf_scores, ar_scores):
        tf_loss = -torch.mean(tf_scores)
        ar_loss = torch.mean(ar_scores)
        return tf_loss, ar_loss

def _calculate_generator_loss(ar_scores):
        """
        Calculates Fool-The-Discriminator loss
        Optionally calculate the reverse loss
        :param tf_scores: Teacher Forcing scores
        :param ar_scores: AutoRegressive scores
        :return:
        """
        loss = -torch.mean(ar_scores)
        return loss

def _need_update(tf_scores, ar_scores):
        """
        Discriminator accuracy < 0.75 --> don't backpropagate to generator
        Discriminator accuracy > 0.99 --> don't train discriminator
        Discriminator guess is calculated as x > 0.5
        :param tf_scores: Teacher Forcing scores [batch_size * 1]
        :param ar_scores: AutoRegressive scores  [batch_size * 1]
        :return:
        """
        correct = float((tf_scores.view(-1) > 0.5).sum() + (ar_scores.view(-1) < 0.5).sum())
        d_accuracy = correct / (tf_scores.size(0) * 2)
        if d_accuracy < 0.75:
            return False, True
        elif d_accuracy > 0.99:
            return True, False
        else:
            return True, True
def loss_mse(x_all, x_all_flip):
        loss = 0
        for i in range(x_all.size(1)):
            loss = loss + F.mse_loss(x_all[:, i], x_all_flip[:, i])
        return loss        


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


def regulization(topic, batchsize):

  topic = topic.view(batchsize, 6, -1)
  
  loss = 0
  num=0
  for i in range(6):
    for j in range(6):
      if i!=j:
        loss = loss + 1 - F.cosine_similarity(topic[:, i, :], topic[:, j, :], dim = -1)
        num+=1
  loss = loss/num
  return torch.sum(loss, 0)/batchsize

def regulization_batch(topic, batchsize):

  topic = topic.view(batchsize, -1)
  
  loss = 0
  num = 0
  for i in range(batchsize):
    for j in range(batchsize):
      if i!=j:
        loss = loss + 1 - F.cosine_similarity(topic[i, :], topic[j, :], dim = -1)
        num+=1
  return loss/num



try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):

    # Load data
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    # Tensorboard summaries (they're great!)
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    # Load pretrained model, info file, histories file
    infos = {}
    histories = {}

    if opt.start_from is not None:
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    #ss_prob_history = histories.get('ss_prob_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    models = Seq2Seq().cuda()
    # Create model

    optimizer = utils.build_optimizer_adam(models.parameters(), opt)

    update_lr_flag = True
    sc_flag = False

    while True:

        # Update learning rate once per epoch
        if update_lr_flag:

            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)

            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                #opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                #model.ss_prob = opt.ss_pro
                
        # Load data from train split (0)
        start = time.time()
        data = loader.get_batch('train')
        data_time = time.time() - start
        start = time.time()

        # Unpack data
        torch.cuda.synchronize()
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['dist'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, dist_label, masks, attmasks = tmp
        labels = labels.long()
        nd_labels = labels
        batchsize = fc_feats.size(0)
        # Forward pass and loss
        d_steps = 1
        g_steps = 1
        #print (torch.sum(labels!=0), torch.sum(masks!=0))
        if 1:
          if 1:
              models.train()
              optimizer.zero_grad()
              wordact = models(labels.view(batchsize, -1).transpose(1,0), labels.view(batchsize, -1).transpose(1,0), fc_feats)
              wordact_t = wordact.transpose(1,0)[:, 1:, :]
              wordact_t = wordact_t.contiguous().view(wordact_t.size(0)*wordact_t.size(1), -1)
              labels_flat = labels.view(batchsize,-1)
              wordclass_v = labels_flat[:, 1:]
              wordclass_t = wordclass_v.contiguous().view(\
               wordclass_v.size(0)*wordclass_v.size(1), -1)
              loss_xe = F.cross_entropy(wordact_t[...], wordclass_t[...].view(-1))

              train_loss = loss_xe
              train_loss.backward()
              optimizer.step()
          
          if 1:
            if iteration % opt.print_freq == 1:
              print('Read data:', time.time() - start)
              if not sc_flag:
                  print("iter {} (epoch {}), train_loss = {:.3f}, data_time = {:.3f}" \
                    .format(iteration, epoch, loss_xe, data_time))
              else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, data_time = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, np.mean(reward[:,0]), data_time, total_time))

          # Update the iteration and epoch
          iteration += 1
          if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

          # Write the training loss summary
          if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            #add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)
            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            #ss_prob_history[iteration] = model.ss_prob

        # Validate and save model 
          if (iteration % opt.save_checkpoint_every == 0):
            checkpoint_path = os.path.join(opt.checkpoint_path, 'langmodel{:05d}.pth'.format(iteration))
            torch.save(models.state_dict(), checkpoint_path)

            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)
            # Evaluate model

       

opt = opts.parse_opt()
opt.batch_size = 13
opt.input_att_dir =  'data/parabu_att'
opt.input_fc_dir = 'data/parabu_fc'
opt.input_json = 'data/paratalk.json'
opt.input_label_h5 = 'data/paratalk_label.h5'
opt.language_eval =1
opt.learning_rate = 4e-4
opt.learning_rate_decay_start =0
opt.scheduled_sampling_start =0
opt.max_epochs= 80
opt.save_checkpoint_every = 2000
opt.checkpoint_path= 'log_cvpr/'
opt.id ='xe'
opt.print_freq =100
opt.model = ''
train(opt)

