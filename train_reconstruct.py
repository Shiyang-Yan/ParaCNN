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
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

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

    # Create model
    model = models.setup(opt).cuda()
    #pretrained_dict = torch.load(opt.model)
    #model.load_state_dict(pretrained_dict, strict=False)

    num_params = get_n_params(model)
    print ('number of parameteres:', num_params)




    dp_model = torch.nn.DataParallel(model)
    dp_model.train()

    # Loss function
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    # Optimizer and learning rate adjustment flag
    optimizer = utils.build_optimizer(model.parameters(), opt)
    update_lr_flag = True

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    # Training loop
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
        fc_feats, att_feats, labels, dist_label, masks, att_masks = tmp
        batchsize = fc_feats.size(0)
        # Forward pass and loss
        optimizer.zero_grad()
        if not sc_flag:
            wordact, reconstruct = dp_model(fc_feats, att_feats, labels)
            #loss_dist = F.binary_cross_entropy(dist, dist_label.cpu().float())
            fc_feats_max, _= att_feats.max(1)
            loss_rec = F.mse_loss(reconstruct.cpu(), fc_feats_max.cpu())
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
            loss = 5 * loss_xe + loss_rec
        else:
            gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())
       
        # Backward pass
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.item()
        torch.cuda.synchronize()

        # Print 
        total_time = time.time() - start
        if  iteration % opt.print_freq == 1:
            print('Read data:', time.time() - start)
            if not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, data_time = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, data_time, total_time))
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
        if (iteration >= 60000 and iteration % opt.save_checkpoint_every == 0):
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)
            # Evaluate model
            eval_kwargs = {'split': 'test',
                            'dataset': opt.input_json}
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

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
