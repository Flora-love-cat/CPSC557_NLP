import torch
import torch.nn as nn
from torchtext.datasets import multi30k, Multi30k
from torch.utils.data import DataLoader
import spacy
from data import token_vocab_transform, collate_fn, PAD_IDX
import functools
from modelTransformer import Encoder, Decoder, Seq2Seq, EncoderLayer, DecoderLayer, SelfAttention, PositionwiseFeedforward, NoamOpt
from train_transformer import train_model 

#=====================================================
# load data
#=====================================================

# Issue: https://github.com/pytorch/text/issues/1756
# Update URLs to point to data stored by user
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"

# Update hash since there is a discrepancy between user hosted test split and that of the test split in the original dataset 
multi30k.MD5["test"] = "6d1ca1dba99e2c5dd54cae1226ff11c2551e6ce63527ebb072a1f70f72a5cd36"

#=====================================================
# Build vocab
#=====================================================

SRC = 'de'  # source language German
TRG = 'en'  # target langauge English 
LANG = {SRC: 0, TRG: 1}
models = ['de_core_news_sm', 'en_core_web_sm'] # Spacy langauge model

train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), 
                                             language_pair=(SRC, TRG))


print(f"Number of training examples: {len(list(train_iter))}")
print(f"Number of validation examples: {len(list(valid_iter))}")
# print(f"Number of testing examples: {len(list(test_iter))}")


token_transform, vocab_transform = token_vocab_transform(LANG, models, train_iter)

print(f"Unique tokens in source (de) vocabulary: {len(vocab_transform[SRC])}")
print(f"Unique tokens in target (en) vocabulary: {len(vocab_transform[TRG])}")

#=====================================================
# Make train/valid/test data loader
#=====================================================
collate_fn_with_vocab = functools.partial(collate_fn, vocab_transform=vocab_transform, token_transform=token_transform, 
                                          SRC=SRC, TRG=TRG, batch_first=True)  # batch first

BATCH_SIZE = 128
train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_vocab)
valid_loader = DataLoader(valid_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_vocab)
test_loader = DataLoader(test_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_vocab)

#=====================================================
# Set hyperparameters
#=====================================================
input_dim = len(vocab_transform[SRC])
output_dim = len(vocab_transform[TRG])
hid_dim = 512
n_layers = 6
n_heads = 8
pf_dim = 2048
dropout = 0.1

N_EPOCHS = 5 
CLIP = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=====================================================
# Option 1: Initialize Transformer model
#=====================================================
enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
model = Seq2Seq(enc, dec, PAD_IDX, device).to(device)

#=====================================================
# Initialize optimizer and loss function
#=====================================================
optimizer = NoamOpt(hid_dim, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)


#=====================================================
# training and evaluation
#=====================================================

train_model(device, model, train_loader, valid_loader, test_loader, 
            optimizer, criterion, N_EPOCHS = N_EPOCHS, CLIP = CLIP, filename='transformer.pt')


