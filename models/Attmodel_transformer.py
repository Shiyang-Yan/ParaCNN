#= This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import math
from .CaptionModel import CaptionModel
import numpy as np
import pdb
def Conv1d(in_channels, out_channels, kernel_size, padding, dropout=0):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

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
         return context



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


##############################################################################

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    query = query.squeeze()
    key = key.squeeze()
    value = value.squeeze()
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    #if mask is not None:
    #    scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Embeddings(nn.Module):
        def __init__(self, d_model, vocab):
            super(Embeddings, self).__init__()
            self.lut = nn.Embedding(vocab, d_model)
            self.d_model = d_model
        def forward(self, x):
            return self.lut(x) * math.sqrt(self.d_model)



################################################################################


class Transformer(nn.Module):
    def clip_att(self, att_feats, att_masks):
                # Clip the length of att_masks and att_feats to the maximum length
         if att_masks is not None:
             max_len = att_masks.data.long().sum(1).max()
             att_feats = att_feats[:,:max_len].contiguous()
             att_masks = att_masks[:,:max_len].contiguous()
         return att_feats, att_masks
    def make_model(self, src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=512, h=1, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        
        model = EncoderDecoder(
             Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
             Decoder(DecoderLayer(d_model, c(attn), c(attn),
             c(ff), dropout), N),
             lambda x:x,
             nn.Sequential(self.embedding, c(position)),
             Generator(d_model, tgt_vocab))
                
        
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model


    def __init__(self):
        super(Transformer, self).__init__()
        # self.config = yaml.load(open(opt.config_file))
        # d_model = self.input_encoding_size # 512
        self.embedding = Embeddings(512, 8668)
        self.use_bn = True
        self.att_feat_size = 512
        self.input_encoding_size = 512
        self.drop_prob_lm = 0.1
        self.vocab_size = 8668
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.input_encoding_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn==2 else ())))

        self.embed = lambda x : x
        self.fc_embed = lambda x : x

        tgt_vocab = self.vocab_size 
        self.model = self.make_model(0, tgt_vocab,
            N=5,
            d_model=self.input_encoding_size,
            d_ff= 512)

    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[...,:1], att_feats[...,:1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)


        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            # crop the last one
            seq_mask = (seq.data > 0)
            seq_mask[:,0] = 1 # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = self.repeat_tensors(seq_per_img,
                    att_feats, att_masks
                )
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq,  att_masks=None, ):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)

        out = self.model(att_feats, seq, att_masks, seq_mask)

        outputs = self.model.generator(out)
        return outputs, out
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0].unsqueeze(1).float(), it.float()], dim=1)
        out = self.model.decode(memory, mask,
                               ys,
                               subsequent_mask(ys.size(1))
                                        .to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]





class convcap(nn.Module):

  def __init__(self, opt):
    super(convcap, self).__init__()
    self.transformer = Transformer()
    self.emb = self.transformer.embedding
    num_wordclass = opt.vocab_size + 1
    self.fusion_feat = Linear(1024, 512)
    num_layers = opt.num_layers
    is_attention = True
    nfeats = 1024
    dropout = .1
    self.fc_imgfeats = Linear(2048, nfeats//2)
    self.nimgfeats = 2048
    self.is_attention = is_attention
    self.nfeats = nfeats
    self.dropout = dropout 
    self.attention_context = AttentionLayer_context()
    self.topic_proj = Linear(self.nfeats, self.nfeats//2)
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
      if(self.is_attention and i%2==0):
        self.attention.append(AttentionLayer())
      n_in = n_out
    self.kernel_topic = 5
    self.pad_topic = self.kernel_topic -1
    self.convs_topic = nn.ModuleList()
    for i in range(5):
      self.convs_topic.append(Conv1d(n_in//2, 2*n_out//2, self.kernel_topic, self.pad_topic))
      n_in = n_out

    self.classifier_0 = Linear(self.nfeats//2, (nfeats // 2))
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
          x = attn(x, wordemb, imgsfeats)
          x = F.relu(x.transpose(2, 1))
    
        x = (x+residual)*math.sqrt(.5)
    return x

  def forward(self, non_used, imgsfeats, wordclass, length, num, attmasks):

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
      
      x_all_para = torch.mean(x_all, 1)
    
      x_all_para = x_all_para.transpose(2,1).cuda()

      x_tobe_conv = torch.cat([x_all_para,para_level_visual], 1)
   
      x_tobe_conv = F.relu(self.topic_proj(x_tobe_conv.transpose(2,1)))
      topic_conv = self.conv_topic(x_tobe_conv.transpose(2,1))

      topic = F.relu(topic_conv[:,:,index])

      wordemb = self.emb(wordclass[:,index,:].squeeze())

      if len(wordemb.size())>2:
        x = wordemb.transpose(2, 1)
        batchsize, wordembdim, maxtokens = x.size()
      else:
        x = wordemb.transpose(1, 0).unsqueeze(0)
        batchsize, wordembdim, maxtokens = x.size()

      y = F.relu(self.imgproj(topic))

      y = y.unsqueeze(2).expand(batchsize, self.nfeats//2, maxtokens)
      x = torch.cat([x, y], 1)
      #fusion = x+y+imgsfeats_used.transpose(2,1)[:,:,:maxtokensi
      imgsfeats_used = imgsfeats_used.transpose(2,1)[:,:,:maxtokens].transpose(2,1)
      fusion = F.relu(self.fusion_feat(x.transpose(2,1)))
      fusion_final = fusion
      x1, out = self.transformer._forward(fusion_final, fusion_final,  wordclass[:, index, :], None)
      x1 = x1.transpose(2,1)

      x2 = F.dropout(out, p=self.dropout, training=self.training)
      x_all_out[:,:,index,:] = out

      output[:, index, :, :] = x1
    if len(output.size())>2:
        output = output.transpose(2,1).contiguous()
    else:
        output = output.transpose(1,0).unsqueeze(0)
    output = output.view((batchsize, 8668, -1))
    x_all_out = x_all_out.permute(0,3,2,1).contiguous()
    x_all_out = x_all_out.view((batchsize, 512, -1)).contiguous()
    return output, x_all_out



