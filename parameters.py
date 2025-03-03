import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'
sourceTestFileName = 'en_bg_data/test.en'
targetTestFileName = 'en_bg_data/test.bg'

ind2tokenFileName = "model/tokenizer/ind2token.pkl"
token2indFileName = "model/tokenizer/token2ind.pkl"
mergesFileName = "model/tokenizer/merges.pkl"

trainFileName = 'data/trainData.pkl'
validationFileName = 'data/validationData.pkl'
testFileName = 'data/testData.pkl'

modelFileName = 'model/NMTmodel'

device = torch.device("cuda:0")
#device = torch.device("cpu")

vocab_size = 8_000
max_seq_len = 1532
num_blocks = 5
d_model = 512
num_heads = 8
d_k = 64 # it is not really used
d_v = d_model // num_heads # it is not really used
dropout = 0.1
d_ff = 1536
label_smoothing = 0.1


learning_rate = 0.001
batch_size = 16
clip_grad = 5.0
max_epochs = 10
warmup_steps = 4000

log_every = 10
test_every = 2000

temperature = 0.2
