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

import pdb

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # outputs = []
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, state)
            outputs[:, i] = output
            # outputs.append(output)

        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k+1].expand(*((beam_size,)+att_masks.size()[1:])).contiguous() if att_masks is not None else None

            it = fc_feats.new_zeros([beam_size], dtype=torch.long) # input <bos>
            logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        # BEG MODIFIED
        trigrams = [] # will be a list of batch_size dictionaries
        # END MODIFIED

        # seq = []
        # seqLogprobs = []
        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:,t-1] = it
                # seq.append(it) #seq[t] the input of t+2 time step

                # seqLogprobs.append(sampleLogprobs.view(-1))
                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)

            logprobs, state = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, state)

            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp


            # BEG MODIFIED ----------------------------------------------------

            # Mess with trigrams
            if block_trigrams and t >= 3 and sample_max:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False) # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)


                # Length penalty
                #penalty = 1.00
                #formula = penalty**t  # (5 + t)**penalty / (5 + 1)**penalty
                #helper = (torch.ones(logprobs.shape) - (1.0 - formula) * (torch.arange(logprobs.shape[1]).expand(logprobs.shape) <= 1).float()) 
                #logprobs = logprobs * helper 
                
            # END MODIFIED ----------------------------------------------------

        return seq, seqLogprobs
        # return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

class AdaAtt_lstm(nn.Module):
    def __init__(self, opt, use_maxout=True):
        super(AdaAtt_lstm, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_maxout = use_maxout

        # Build a LSTM
        self.w2h = nn.Linear(self.input_encoding_size, (4+(use_maxout==True)) * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size)

        self.i2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers - 1)])
        self.h2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers)])

        # Layers for getting the fake region
        if self.num_layers == 1:
            self.r_w2h = nn.Linear(self.input_encoding_size, self.rnn_size)
            self.r_v2h = nn.Linear(self.rnn_size, self.rnn_size)
        else:
            self.r_i2h = nn.Linear(self.rnn_size, self.rnn_size)
        self.r_h2h = nn.Linear(self.rnn_size, self.rnn_size)


    def forward(self, xt, img_fc, state):

        hs = []
        cs = []
        for L in range(self.num_layers):
            # c,h from previous timesteps
            prev_h = state[0][L]
            prev_c = state[1][L]
            # the input to this layer
            if L == 0:
                x = xt
                i2h = self.w2h(x) + self.v2h(img_fc)
            else:
                x = hs[-1]
                x = F.dropout(x, self.drop_prob_lm, self.training)
                i2h = self.i2h[L-1](x)

            all_input_sums = i2h+self.h2h[L](prev_h)

            sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
            sigmoid_chunk = F.sigmoid(sigmoid_chunk)
            # decode the gates
            in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
            forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
            out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
            # decode the write inputs
            if not self.use_maxout:
                in_transform = F.tanh(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size))
            else:
                in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
                in_transform = torch.max(\
                    in_transform.narrow(1, 0, self.rnn_size),
                    in_transform.narrow(1, self.rnn_size, self.rnn_size))
            # perform the LSTM update
            next_c = forget_gate * prev_c + in_gate * in_transform
            # gated cells form the output
            tanh_nex_c = F.tanh(next_c)
            next_h = out_gate * tanh_nex_c
            if L == self.num_layers-1:
                if L == 0:
                    i2h = self.r_w2h(x) + self.r_v2h(img_fc)
                else:
                    i2h = self.r_i2h(x)
                n5 = i2h+self.r_h2h(prev_h)
                fake_region = F.sigmoid(n5) * tanh_nex_c

            cs.append(next_c)
            hs.append(next_h)

        # set up the decoder
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)

        state = (torch.cat([_.unsqueeze(0) for _ in hs], 0), 
                torch.cat([_.unsqueeze(0) for _ in cs], 0))
        return top_h, fake_region, state

class AdaAtt_attention(nn.Module):
    def __init__(self, opt):
        super(AdaAtt_attention, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size

        # fake region embed
        self.fr_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.ReLU(), 
            nn.Dropout(self.drop_prob_lm))
        self.fr_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        # h out embed
        self.ho_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.Tanh(), 
            nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed, att_masks=None):

        # View into three dimensions
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)

        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)

        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)

        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))

        img_all = torch.cat([fake_region.view(-1,1,self.input_encoding_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1,1,self.input_encoding_size), conv_feat_embed], 1)

        hA = F.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA,self.drop_prob_lm, self.training)
        
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, att_size + 1), dim=1)

        if att_masks is not None:
            att_masks = att_masks.view(-1, att_size)
            PI = PI * torch.cat([att_masks[:,:1], att_masks], 1) # assume one one at the first time step.
            PI = PI / PI.sum(1, keepdim=True)

        visAtt = torch.bmm(PI.unsqueeze(1), img_all)
        visAttdim = visAtt.squeeze(1)

        atten_out = visAttdim + h_out_linear

        h = F.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h

class AdaAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(AdaAttCore, self).__init__()
        self.lstm = AdaAtt_lstm(opt, use_maxout)
        self.attention = AdaAtt_attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        h_out, p_out, state = self.lstm(xt, fc_feats, state)
        atten_out = self.attention(h_out, p_out, att_feats, p_att_feats, att_masks)
        return atten_out, state

class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

################################################################################

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

class AttentionLayer(nn.Module):
  def __init__(self, conv_channels, embed_dim):
    super(AttentionLayer, self).__init__()
    self.in_projection = Linear(conv_channels, embed_dim)
    self.out_projection = Linear(embed_dim, conv_channels)
    self.bmm = torch.bmm

  def forward(self, x, wordemb, imgsfeats):
    residual = x

    x = (self.in_projection(x) + wordemb) * math.sqrt(0.5)

    b, c, n = imgsfeats.size()
    y = imgsfeats.transpose(2, 1)

    x = self.bmm(x, y)

    sz = x.size()
    x = F.softmax(x.view(sz[0] * sz[1], sz[2]))
    x = x.view(sz)
    attn_scores = x

    y = y.permute(0, 2, 1)
    x = self.bmm(x, y)

    s = y.size(1)
    x = x * (s * math.sqrt(1.0 / s))

    x = (self.out_projection(x) + residual) * math.sqrt(0.5)

    return x, attn_scores

class convcap(nn.Module):
  
  def __init__(self, opt):
    super(convcap, self).__init__()
    num_wordclass = opt.vocab_size + 1
    num_layers = opt.num_layers
    is_attention = False
    nfeats = 1024
    dropout = .1
    self.fc_vgg16= Linear(4096, 512)
    self.fc_imgfeats = Linear(2048, 512)
    self.nimgfeats = 2048
    self.is_attention = is_attention
    self.nfeats = nfeats
    self.dropout = dropout 
    
    self.emb_0 = Embedding(num_wordclass, nfeats//2, padding_idx=0)
    self.emb_1 = Linear(nfeats//2, nfeats//2, dropout=dropout)
    self.topic_proj = Linear(self.nfeats + self.nfeats//2, self.nfeats//2)
    self.imgproj = Linear(self.nfeats//2, self.nfeats//2, dropout=dropout)
    self.resproj = Linear(nfeats, self.nfeats//2, dropout=dropout)
    self.resproj_para = Linear(nfeats//2, self.nfeats//2 ,dropout=dropout)
    n_in = 2*self.nfeats
    n_out = self.nfeats
    self.n_layers = num_layers
    self.convs = nn.ModuleList()
    self.attention = nn.ModuleList()
    self.kernel_size = 5
    self.pad = self.kernel_size - 1
    for i in range(self.n_layers):
      self.convs.append(Conv1d(n_in//2, 2*n_out//2, self.kernel_size, self.pad, dropout))
      if(self.is_attention and i%4==0):
        self.attention.append(AttentionLayer(n_out, nfeats))
      n_in = n_out
    self.kernel_topic = 5
    self.pad_topic = self.kernel_topic -1
    self.convs_topic = nn.ModuleList()
    for i in range(3):
      self.convs_topic.append(Conv1d(n_in//2, 2*n_out//2, self.kernel_topic, self.pad_topic))
      n_in = n_out

    self.classifier_0 = Linear(self.nfeats//2, (nfeats // 2))
    #self.bn1 = nn.BatchNorm1d(num_features = (nfeats // 2))
    self.classifier_1 = Linear((nfeats // 2), num_wordclass, dropout=dropout)


    #self.fc1 = Linear(512, 512)
    #self.fc2 = Linear(512, 1)


  def conv_topic(self, x_all):
    for i, conv_topic in enumerate(self.convs_topic):
      
        if(i==0):
          x_all=x_all.transpose(2,1)
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

  def conv_cap(self, x, wordemb):
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

        if(self.is_attention and i%4 == 0):
          attn = self.attention[int(i/4)]
          x = x.transpose(2, 1)
          x, attn_buffer = attn(x, wordemb, imgsfeats)
          x = x.transpose(2, 1)
    
        x = (x+residual)*math.sqrt(.5)
    return x

  def forward(self, vgg16, fc, imgsfeats, wordclass):
    if 1:
      imgsfeats_used = F.relu(self.fc_imgfeats(imgsfeats)) 
      imgsfeat_vgg16 = F.relu(self.fc_vgg16(vgg16))
      attn_buffer = None
      wordclass = wordclass
      imgsfc7,_ = imgsfeats_used.max(1)
      imgsfc7 = torch.cat((imgsfeat_vgg16, imgsfc7), 1)
      batchsize = imgsfc7.size(0)
      output = torch.zeros((batchsize, 6, 8668, 30))
      para_level_visual = imgsfc7.unsqueeze(2).expand(imgsfc7.size(0), self.nfeats, 6)
      dis_all = torch.zeros((batchsize, 6))
      x_all = torch.zeros((batchsize, 30, 6, 512)).cuda()
      x_all_output = torch.zeros((batchsize, 30, 6, 512)).cuda()
      for index in range(6):
        if index >0:
          x_all[:, :, index, :] = x2
      
        x_all_para = torch.mean(x_all, 1)
    
        x_all_para = x_all_para.transpose(2,1)
      
        x_tobe_conv = torch.cat([x_all_para,para_level_visual], 1)
   
        x_tobe_conv = F.relu(self.topic_proj(x_tobe_conv.transpose(2,1)))
        topic_conv = self.conv_topic(x_tobe_conv.transpose(2,1))
        

        wordemb = self.emb_0(wordclass[:,index,:].squeeze())
        wordemb = self.emb_1(wordemb)
        if len(wordemb.size())>2:
          x = wordemb.transpose(2, 1)  
          batchsize, wordembdim, maxtokens = x.size()
        else:
          x = wordemb.transpose(1, 0).unsqueeze(0)
          batchsize, wordembdim, maxtokens = x.size()
      
        topic = F.relu(topic_conv[:,:,index])


        y = F.relu(self.imgproj(topic))
      
        y = y.unsqueeze(2).expand(batchsize, self.nfeats//2, maxtokens)
        x = torch.cat([x, y], 1)

        x = self.conv_cap(x, wordemb)

        x = x.transpose(2, 1)
     
        x = self.classifier_0(x)

        x = x.transpose(2, 1)

        x1 = F.relu(x)
      
        x1 = x1.transpose(2,1)

        x2 = F.dropout(x1, p=self.dropout, training=self.training)
        x_all_output[:, :, index, :] = x2

        x3 = self.classifier_1(x2)
        x3 = x3.transpose(2, 1)

        output[:, index,:] = x3
      if len(output.size())>2:
            output = output.transpose(2,1).contiguous()
      else:
            output = output.transpose(1,0).unsqueeze(0)
      output = output.view((batchsize, 8668, -1))
      x_all_output = x_all_output.permute(0,3,2,1).contiguous()
      x_all_output = x_all_output.view((batchsize, 512, -1)).contiguous()
    return output, x_all_output

###############################################################################








############################################################################
# Notice:
# StackAtt and DenseAtt are models that I randomly designed.
# They are not related to any paper.
############################################################################

#from .FCModel import LSTMCore
class StackAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(StackAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        # self.att0 = Attention(opt)
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt) # att_feat + word_embedding
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        # self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt,fc_feats],1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0,att_res_1],1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([h_1,att_res_2],1), [state[0][2:3], state[1][2:3]])

        return h_2, [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]

class DenseAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(DenseAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        # self.att0 = Attention(opt)
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt) # att_feat + word_embedding
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        # self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        # fuse h_0 and h_1
        self.fusion1 = nn.Sequential(nn.Linear(opt.rnn_size*2, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))
        # fuse h_0, h_1 and h_2
        self.fusion2 = nn.Sequential(nn.Linear(opt.rnn_size*3, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt,fc_feats],1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0,att_res_1],1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([self.fusion1(torch.cat([h_0, h_1], 1)),att_res_2],1), [state[0][2:3], state[1][2:3]])

        return self.fusion2(torch.cat([h_0, h_1, h_2], 1)), [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res


class Att2in2Core(nn.Module):
    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
            self.a2c(att_res)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state


"""
Note this is my attempt to replicate att2all model in self-critical paper.
However, this is not a correct replication actually. Will fix it.
"""
class Att2all2Core(nn.Module):
    def __init__(self, opt):
        super(Att2all2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        self.a2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1]) + self.a2h(att_res)
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class AdaAttModel(AttModel):
    def __init__(self, opt):
        super(AdaAttModel, self).__init__(opt)
        self.core = AdaAttCore(opt)

# AdaAtt with maxout lstm
class AdaAttMOModel(AttModel):
    def __init__(self, opt):
        super(AdaAttMOModel, self).__init__(opt)
        self.core = AdaAttCore(opt, True)

class Att2in2Model(AttModel):
    def __init__(self, opt):
        super(Att2in2Model, self).__init__(opt)
        self.core = Att2in2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x

class Att2all2Model(AttModel):
    def __init__(self, opt):
        super(Att2all2Model, self).__init__(opt)
        self.core = Att2all2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
'''
class TopDownModel(nn.Module):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = convcap(opt)
'''
class StackAttModel(AttModel):
    def __init__(self, opt):
        super(StackAttModel, self).__init__(opt)
        self.num_layers = 3
        self.core = StackAttCore(opt)

class DenseAttModel(AttModel):
    def __init__(self, opt):
        super(DenseAttModel, self).__init__(opt)
        self.num_layers = 3
        self.core = DenseAttCore(opt)
