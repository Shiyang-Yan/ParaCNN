from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
from torch.autograd import Variable
from matplotlib import pyplot as plt
import matplotlib as mpl
def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/para_captions_test.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out
def plot_kernels(tensor, j, num_cols=5):
    print (tensor.shape)
    ix = 1
    for i in range(6):
        n_filters = tensor[:, i, :]
        # specify subplot and turn of axis
        ax = plt.subplot(1, 1, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(np.reshape(n_filters[:5,:], [5,5]), cmap='gray')
        plt.savefig('foo' + str(j) + '.png')
def showAttention(input_sentence, attentions):
    # Set up figure with colorbar
    x_axis = input_sentence.split(' ')
    leng = len(x_axis) - 1
    fig = plt.figure()
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=15)
    cax = ax.matshow(attentions[:leng,:leng], cmap=cmap, norm=norm)
    cb = fig.colorbar(cax)
    cb.ax.tick_params(labelsize=15)

    ax.set_xticklabels(x_axis, rotation=90)
    ax.set_yticklabels(x_axis)

    # Show label at every tick
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))

    plt.show()
    plt.savefig('foo' + '.png')

def eval_split(model, crit,loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

        
        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * 1], 
            data['att_feats'][np.arange(loader.batch_size) * 1],
            data['labels'][np.arange(loader.batch_size) * 1],
            data['att_masks'][np.arange(loader.batch_size) * 1] if data['att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats,labels, att_masks = tmp
        labels = labels.view(labels.size(0), -1)
        # forward the model to also get generated samples for each image
        max_tokens = 180
        trigrams = []
        self_alpha_all =[]
        seq = torch.zeros((fc_feats.size(0), max_tokens))
        with torch.no_grad():
            wordclass_feed = np.zeros((fc_feats.size(0), max_tokens), dtype='int64') 
            outcaps = np.empty((fc_feats.size(0), 0)).tolist() 
            wordclass_feed[:, 0] = 8667
            kit = 0
            for j in range(max_tokens-1):
                if j>0 and j%30==0:
                    wordclass_feed[:, j] = 8667
                    '''
                    for name, param in model.named_parameters():
                        # print(name)
                        if 'convs_topic.3.weight_v' in name:
                            tensor = param.data.cpu().numpy()
                            # print('plot')
                            # print(tensor)
                            plot_kernels(tensor, kit)
                    kit = kit + 1 
                    '''
                    kit = kit + 1
                    self_alpha_current = self_alpha[:, kit, 2, :, :].squeeze().cpu().numpy()
                    self_alpha_all.append(self_alpha_current)

                if j==0:
                    wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()
                    wordclass = wordclass.view((wordclass.size(0), 6, 30))
                    wordact, _, self_alpha = model(fc_feats, att_feats, wordclass.long(), 30, 6)
                    wordact = wordact[:,:,:-1]
                    wordact_t = wordact.contiguous().transpose(2,1).contiguous().view(fc_feats.size(0)*(max_tokens-1), -1)
                    wordprobs = F.log_softmax(wordact_t, -1).cpu().numpy()
                    wordids = np.argmax((wordprobs), 1)
                    wordids = np.reshape(wordids,(fc_feats.size(0), max_tokens-1))
                if j >=1:
                    #if j == 2:
                    #    unfinished = wordids[:, j] > 0
                    #else:
                    #    unfinished = unfinished * (wordids[:, j] > 0)
                    #if unfinished.sum() == 0:
                    #    break
                    #wordids[:,j] = wordids[:,j]*unfinished.type_as(wordids[:,j]) 
                    seq[:, j-1] = torch.tensor(wordids[:, j])
                    wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()
                    wordclass = wordclass.view((wordclass.size(0), 6, 30))
                    wordact, _, self_alpha = model(fc_feats, att_feats, wordclass, 30, 6)
                    wordact = wordact[:,:,:-1]
                    wordact_t = wordact.contiguous().transpose(2,1).contiguous().view(fc_feats.size(0)*(max_tokens-1), -1)
                    wordprobs = F.log_softmax(wordact_t, -1)
                    wordprobs = wordprobs.view(fc_feats.size(0), max_tokens-1, -1)
                
                if j>= 3:
                     prev_two_batch = seq[:,j-3:j-1]
                     for p in range(fc_feats.size(0)):
                         prev_two = (prev_two_batch[p][0].item(), prev_two_batch[p][1].item())
                         current = seq[p][j-1]
                         if j==3:
                             trigrams.append({prev_two: [current]})
                         elif j>3:
                             if prev_two in trigrams[p]: # add to list
                                 trigrams[p][prev_two].append(current)
                             else: # create list
                                 trigrams[p][prev_two] = [current]

                     prev_two_batch = seq[:,j-2:j]
                     mask = torch.zeros((fc_feats.size(0), max_tokens-1,  8668)).cuda()
                     for p in range(fc_feats.size(0)):
                        prev_two = (prev_two_batch[p][0].item(), prev_two_batch[p][1].item())
                        if prev_two in trigrams[p] and 0. not in prev_two and 8666. not in prev_two and 8667. not in prev_two:
                            for q in trigrams[p][prev_two]:
                                q = int(q)
                                mask[p,:, q] += 1
                     alpha  = 3
                     wordprobs = wordprobs.cuda() + (mask * -0.693 * alpha) # ln(1/2) * a
                if j>=1:
                     wordprobs = wordprobs.view(fc_feats.size(0)*(max_tokens-1), -1).cpu().numpy()
                     wordids = np.argmax(wordprobs, 1)
                     wordids = np.reshape(wordids,(fc_feats.size(0), max_tokens-1))
                for k in range(fc_feats.size(0)):
                    word = wordids[k][j]
                    outcaps[k].append(word)
                    if(j < max_tokens-1):
                        wordclass_feed[k, j+1] = wordids[k][j]
        seq = torch.tensor(outcaps)
        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        prob = 0
        sents = utils.decode_sequence(loader.get_vocab(), seq, prob)

        for k, sent in enumerate(sents):
            sent = sent.replace('<start>', '').replace(' .', '.').replace(' ,', ',').replace('UNK', '').replace('   ', ' ').replace('  ',' ').replace(' \'', '\'')
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if 1:
                print('image %s: %s' %(entry['image_id'], entry['caption']))
            if k == 4:
                for ti in range(5):
                    attention = self_alpha_all[ti]
                    attention = np.reshape(attention[k], [30, 30])
                    #print (attention)
                    caption_all = entry['caption'].split('.')
                    showAttention(caption_all[ti], attention)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break
        
    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats
