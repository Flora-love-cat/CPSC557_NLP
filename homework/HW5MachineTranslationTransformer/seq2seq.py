import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import multi30k, Multi30k
from torch.utils.data import DataLoader
import spacy
from data import token_vocab_transform, collate_fn, PAD_IDX, BOS_IDX, EOS_IDX
import functools
from modelseq2seq import Encoder, Decoder, Seq2Seq, Attention, DecoderAtten, Seq2SeqAtten
from train_seq2seq import train_model 
from inference import translate_sentence, display_attention

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
collate_fn_with_vocab = functools.partial(collate_fn, vocab_transform=vocab_transform, token_transform=token_transform, SRC=SRC, TRG=TRG)

BATCH_SIZE = 128
train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_vocab)
valid_loader = DataLoader(valid_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_vocab)
test_loader = DataLoader(test_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_vocab)

#=====================================================
# Set hyperparameters
#=====================================================
INPUT_DIM = len(vocab_transform[SRC])
OUTPUT_DIM = len(vocab_transform[TRG])
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_NUM_LAYERS = 1
DEC_NUM_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

lr = 0.001
N_EPOCHS = 5 
CLIP = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=====================================================
# Option 1: Initialize vanilla Seq2Seq model
#=====================================================
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_NUM_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, DEC_NUM_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

#=====================================================
# Option 2: Initialize vanilla Seq2Seq model
#=====================================================

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_NUM_LAYERS, ENC_DROPOUT)
dec = DecoderAtten(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_NUM_LAYERS, DEC_DROPOUT, attn)
model = Seq2SeqAtten(enc, dec, PAD_IDX, BOS_IDX, EOS_IDX, device).to(device)

#=====================================================
# Initialize optimizer and loss function
#=====================================================
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)


#=====================================================
# training and evaluation
#=====================================================

train_model(device, model, train_loader, valid_loader, test_loader, 
            optimizer, criterion, N_EPOCHS = N_EPOCHS, CLIP = CLIP, filename='seq2seq.pt')


#=====================================================
# inference 
#=====================================================
source = "Die Katze wurde vom Hund gejagt"
# target = "the cat was chased by the dog"

translation, attention = translate_sentence(model, source, token_transform, vocab_transform, SRC, TRG)
display_attention(source, translation, attention, token_transform, SRC)