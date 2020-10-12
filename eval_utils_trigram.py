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

def eval_split(model, crit, loader, eval_kwargs={}):
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
            data['att_masks'][np.arange(loader.batch_size) * 1] if data['att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        # forward the model to also get generated samples for each image
        max_tokens = 30*6
        trigrams = []
        with torch.no_grad():
            wordclass_feed = np.zeros((fc_feats.size(0), max_tokens), dtype='int64') 
            outcaps = np.empty((fc_feats.size(0), 0)).tolist() 
            wordclass_feed[:, 0] = 8667
            for j in range(max_tokens-1):
              if j % 30 ==0:
                  wordclass_feed[:, j] = 8667
              wordclass_feed_used = np.reshape(wordclass_feed, (fc_feats.size(0), 6, 30))
              wordclass = Variable(torch.from_numpy(wordclass_feed_used)).cuda()
              wordact, _, = model(fc_feats, att_feats, wordclass, 30, 6)
              wordact = wordact[:,:,:-1]
              wordact_t = wordact.contiguous().view(fc_feats.size(0), -1, (max_tokens-1))
              wordprobs = F.log_softmax(wordact_t, 1).cuda()
              wordids = torch.argmax(wordprobs, 1)
              wordids_rshp = wordids.view((fc_feats.size(0), max_tokens-1))
              if j>= 3:
                 prev_two_batch = wordids_rshp[:,j-3:j-1]
                 for p in range(fc_feats.size(0)):
                     prev_two = (prev_two_batch[p][0].item(), prev_two_batch[p][1].item())
                     current  = wordids_rshp[p][j]
                     if j == 3:
                         trigrams.append({prev_two: [current]})
                     else:
                         trigrams[p][prev_two] = [current]

                 prev_two_batch = wordids_rshp[:,j-2:j]
                 mask = torch.zeros((fc_feats.size(0), 8668, max_tokens-1))
                 for p in range(fc_feats.size(0)):
                    prev_two = (prev_two_batch[p][0].item(), prev_two_batch[p][1].item())
                    if prev_two in trigrams[p]:
                        for q in trigrams[p][prev_two]:
                            mask[p,q] += 1
                 alpha = 2.0 # = 4
                 wordprobs = wordprobs.cuda() + (mask.cuda() * -0.693 * alpha) # ln(1/2) * a
              wordids = torch.argmax(wordprobs, 1)
              wordids_rshp = wordids.view((fc_feats.size(0), max_tokens-1))
              for k in range(fc_feats.size(0)):
                 word = wordids_rshp[k][j]
                 outcaps[k].append(word)
                 if(j < max_tokens-1):
                   wordclass_feed[k, j+1] = wordids_rshp[k][j]
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

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

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
