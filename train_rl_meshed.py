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

import opts
from models.transformer_revise import *
import eval_utils
import misc.utils as utils

from sample import *
import misc.utils as utils
from rl_utils import *
from dataloader import *

rl_crit = utils.RewardCriterion()

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

       
def train(opt):
    iteration = 0
    epoch = 0
    # Load data
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    # Tensorboard summaries (they're great!)
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)
    
    # Load pretrained model, info file, histories file
    infos = {}
    histories = {}

    # Create model
    model = transformer.convcap(opt).cuda()
    pretrained_dict = torch.load(opt.model)
    model.load_state_dict(pretrained_dict, strict=False)
    start = time.time()
    dp_model = torch.nn.DataParallel(model)
    dp_model.train()
    
    optimizer = utils.build_optimizer(model.parameters(), opt)
    update_lr_flag = True
    samplenet = sampleNet(dp_model, opt)
    while True:
        # Unpack data
    #torch.cuda.synchronize()
      data = loader.get_batch('train')
      data_time = time.time() - start
      tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['dist'], data['masks'], data['att_masks']]
      tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
      fc_feats, att_feats, labels, dist_label, masks, att_masks = tmp
      batchsize = fc_feats.size(0)
    # Forward pass and loss
      optimizer.zero_grad()
      labels_decode = labels.view(-1, 180)
      captions = utils.decode_sequence(loader.get_vocab(), labels_decode, None)
      captions_all = []
      for index,caption in enumerate(captions):
          caption = caption.replace('<start>', '').replace(' ,', '').replace('  ', ' ')
          captions_all.append(caption)
      #print (captions_all[0])
      #with torch.no_grad():
      target, outcap, sample_right = samplenet(batchsize, 30*6, loader.get_vocab(), att_feats)
      #wordclass_feed = wordclass_feed.reshape((batchsize, 6, 30))
      #out, _ = dp_model(fc_feats, att_feats, torch.tensor(wordclass_feed))
      #Logprobs = torch.log(F.softmax(out.transpose(2,1)))
      #target = target.view((batchsize, (30*6), -1))
      #sampleLogprobs = Logprobs.gather(2, target.long().unsqueeze(2)) # gather t
      #print (sampleLogprobs.size(), sample_right.size())
      #print (sampleLogprobs.squeeze()[:, :], sample_right[:, :])
      with torch.no_grad():
        reward, cider_sample, cider_greedy = get_self_critical_reward(batchsize, dp_model, att_feats, outcap, captions_all, loader.get_vocab(), 30*6)
      loss_rl = rl_crit(sample_right, target.data, torch.from_numpy(reward).float()) 
      wordact, x_all = dp_model(fc_feats, att_feats, labels, 30, 6)
      mask = masks[:,1:].contiguous()
      wordact = wordact[:,:,:-1]
      wordact_t = wordact.permute(0, 2, 1).contiguous()
      wordact_t = wordact_t.view(wordact_t.size(0) * wordact_t.size(1), -1)
      labels = labels.contiguous().view(-1, 6*30).cpu()
      wordclass_v = labels[:,1:]
      wordclass_t = wordclass_v.contiguous().view(\
         wordclass_v.size(0) * wordclass_v.size(1), 1)
      maskids = torch.nonzero(mask.view(-1).cpu()).numpy().reshape(-1)
      loss_xe = F.cross_entropy(wordact_t[maskids, ...], \
         wordclass_t[maskids, ...].contiguous().view(maskids.shape[0])) 
      loss_xe_all = loss_rl #+ F.mse_loss(x_all_inference.cuda(), x_all.cuda()).cuda()
      loss_xe_all.backward()
      utils.clip_gradient(optimizer, opt.grad_clip)
      optimizer.step()
      train_loss = loss_xe_all.item()
      torch.cuda.synchronize()
        # Print 
      total_time = time.time() - start
      reward = reward[:,0].mean()
      cider_sample = cider_sample.mean()
      cider_greedy = cider_greedy.mean()
      if 1:
        if iteration % 2 == 1:
            print('Read data:', time.time() - start)
            if 0:
                print("iter {} (epoch {}), train_loss = {:.3f}, data_time = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, data_time, total_time))
            if 1:
                print("iter {} (epoch {}), train_loss = {:.3f}, avg_reward = {:.3f},cider_sample  = {:.3f}, cider_greedy ={:.3f},  data_time = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, reward, cider_sample, cider_greedy, data_time, total_time))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        '''
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            #add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)
            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            #ss_prob_history[iteration] = model.ss_prob
        '''
        # Validate and save model 
        if (iteration % opt.save_checkpoint_every == 0):
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model'+str(iteration)+'.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)
            # Evaluate model
            '''
            eval_kwargs = {'split': 'test',
                            'dataset': opt.input_json}
            crit = 1
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, crit, loader, eval_kwargs)
            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if lang_stats is not None:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Our metric is CIDEr if available, otherwise validation loss
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            # Save model in checkpoint path 
            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = loader.get_vocab()
            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            #histories['ss_prob_history'] = ss_prob_history
            with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                cPickle.dump(infos, f)
            with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                cPickle.dump(histories, f)

            # Save model to unique file if new best model
            if best_flag:
                model_fname = 'model-best.pth'
                infos_fname = 'model-best.pkl'
                checkpoint_path = os.path.join(opt.checkpoint_path, model_fname)
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                with open(os.path.join(opt.checkpoint_path, infos_fname), 'wb') as f:
                    cPickle.dump(infos, f)
            '''
opt = opts.parse_opt()
opt.batch_size = 13
opt.input_att_dir = 'data/parabu_att'
opt.input_fc_dir = 'data/parabu_fc'
opt.input_json = 'data/paratalk.json'
opt.input_label_h5 = 'data/paratalk_label.h5'
opt.language_eval = 1
opt.learning_rate = 3e-5
opt.learning_rate_decay_start = 0
opt.scheduled_sampling_start = 0
opt.max_epochs = 80
opt.save_checkpoint_every = 500
opt.checkpoint_path = 'log_cvpr2/'
opt.id = 'xe'
opt.print_freq = 100
opt.model = 'eccv_model/all2model25000.pth'


train(opt)





