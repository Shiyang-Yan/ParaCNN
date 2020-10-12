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
from torch.autograd import Variable
from itertools import chain
from functools import reduce
from discriminator import Discriminator
from seq_model import *
from rl_utils import *
from sample import *
import misc.utils as utils
rl_crit = utils.RewardCriterion()

def train(opt):

    # Load data
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    # Tensorboard summaries (they're great!)

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
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(8668, 180, 3, 0)
    models = Transformer(8667, encoder, decoder)
    # Create model
    model = models.cuda()
    lang_model = Seq2Seq().cuda()
    # Create model
    model.load_state_dict(torch.load('./log_cvpr_mesh/all2model20000.pth'))
    lang_model.load_state_dict(torch.load('log_cvpr/all2model16000.pth'), strict=False)
    optimizer = utils.build_optimizer_adam(list(models.parameters()) + list(lang_model.parameters()), opt)

    update_lr_flag = True


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
                #model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False

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
        batchsize = fc_feats.size(0)
        labels_decode = labels.view(-1, 180)
        captions = utils.decode_sequence(loader.get_vocab(), labels_decode, None)
        captions_all = []
        for index, caption in enumerate(captions):
            caption = caption.replace('<start>', '').replace(' ,', '').replace('  ', ' ')
            captions_all.append(caption)

        # Forward pass and loss
        d_steps = 1
        g_steps = 1
        #print (torch.sum(labels!=0), torch.sum(masks!=0))
        if 1:




          if 1:
              model.train()
              optimizer.zero_grad()
              wordact, x_all_image = model(att_feats, labels.view(batchsize, -1))
              wordact_t = wordact[:,:-1,:]
              wordact_t = wordact_t.contiguous().view(wordact_t.size(0) * wordact_t.size(1), -1)
              labels_flat = labels.view(batchsize,-1)
              wordclass_v = labels_flat[:, 1:]
              wordclass_t = wordclass_v.contiguous().view(\
               wordclass_v.size(0) * wordclass_v.size(1), -1)
              loss_xe = F.cross_entropy(wordact_t[ ...], \
               wordclass_t[...].contiguous().view(-1))
              '''
              wordact = lang_model(labels.view(batchsize, -1).transpose(1, 0), labels.view(batchsize, -1).transpose(1, 0),
                               fc_feats)
              wordact_t = wordact.transpose(1, 0)[:, 1:, :]
              wordact_t = wordact_t.contiguous().view(wordact_t.size(0) * wordact_t.size(1), -1)
              labels_flat = labels.view(batchsize, -1)
              wordclass_v = labels_flat[:, 1:]
              wordclass_t = wordclass_v.contiguous().view( \
                  wordclass_v.size(0) * wordclass_v.size(1), -1)
              loss_xe_lang = F.cross_entropy(wordact_t[...], wordclass_t[...].view(-1))
              '''
              outcap, sampled_ids, sample_logprobs= lang_model.sample(labels.view(batchsize, -1).transpose(1,0),labels.view(batchsize, -1).transpose(1,0), fc_feats, loader.get_vocab())
              sampled_ids[:, 0] = 8667
              logprobs_input, _ = model(att_feats, sampled_ids.long().cuda())
              log_probs = F.log_softmax(logprobs_input[:, :-1, :], -1)

              sample_logprobs_true = log_probs.gather(2, sampled_ids[:, 1:].cuda().long().unsqueeze(2))



              with torch.no_grad():
                  reward, cider_sample, cider_greedy = get_self_critical_reward(batchsize, lang_model, labels.view(batchsize, -1).transpose(1,0), fc_feats, outcap,
                                                                                captions_all, loader,
                                                                                180)

              print (np.mean(cider_greedy))
              loss_rl1 = rl_crit(torch.exp(sample_logprobs_true.squeeze()) / torch.exp(sample_logprobs[:, 1:]).cuda().detach(),sampled_ids[:, 1:].cpu(), torch.from_numpy(reward).float().cuda())

              #loss_rl = rl_crit(sample_logprobs, sampled_ids.cpu(), torch.from_numpy(reward).float()).cuda()
              #x_all_langauge = x_all_langauge.cuda().detach()
              #l2_loss = ((x_all_image.transpose(2,1).cuda() - x_all_langauge) ** 2).mean().cuda()
              train_loss = loss_xe + loss_rl1 # + loss_xe_lang
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
            checkpoint_path = os.path.join(opt.checkpoint_path, 'all2model{:05d}.pth'.format(iteration))
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_path = os.path.join(opt.checkpoint_path, 'lang_model{:05d}.pth'.format(iteration))
            torch.save(lang_model.state_dict(), checkpoint_path)
            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)
            # Evaluate model

opt = opts.parse_opt()
opt.batch_size = 10
opt.input_att_dir =  'data/parabu_att'
opt.input_fc_dir = 'data/parabu_fc'
opt.input_json = 'data/paratalk.json'
opt.input_label_h5 = 'data/paratalk_label.h5'
opt.language_eval =1
opt.learning_rate = 0.0004
opt.learning_rate_decay_start =0
opt.scheduled_sampling_start =0
opt.max_epochs= 80
opt.save_checkpoint_every = 2000
opt.checkpoint_path= 'log_cvpr_off/'
opt.id ='xe'
opt.print_freq =100
opt.model = ''
train(opt)

