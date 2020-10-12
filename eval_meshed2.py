from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from dataloader import *
from dataloaderraw import *
import eval_utils_trigram
import misc.utils as utils
import torch
import opts
opt = opts.parse_opt()
from models.transformer_revise.transformer import convcap

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
opt.model = 'log_cvpr2/model500.pth'
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
models = convcap(opt)
models.cuda()
models.load_state_dict(torch.load(opt.model), strict=False)
models.eval()

crit = utils.LanguageModelCriterion()





# Set sample options
loss, split_predictions, lang_stats = eval_utils_trigram.eval_split(models, crit, loader,
    vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
