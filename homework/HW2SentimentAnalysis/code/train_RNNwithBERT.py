import functools
from torch.utils.data import random_split, DataLoader
from data import DataFrameDataset, collate_fn, tokenize_bert
import pandas as pd 
import torch
from model import BERTwithRNN 
from train import train_model 
from transformers import BertTokenizer, BertModel

def main():
    #========================================================
    # load data into Pandas dataframe 
    #========================================================

    data_path = './train.tsv' # TODO: replace this with your own path to the dataset (should end in train.tsv)
    all_raw = pd.read_table(data_path)
    # Step 1: Sort the DataFrame by SentenceId and PhraseId
    all_data = all_raw.sort_values(['SentenceId', 'PhraseId'])

    # Group by SentenceId and keep the first record for each group
    all_data = all_data.groupby('SentenceId').first().reset_index()

    # Drop PhraseId and SentenceId columns
    all_data = all_data.drop(['PhraseId', 'SentenceId'], axis=1)

    # Rename the column 'Phrase' to 'text'
    all_data = all_data.rename(columns={'Phrase': 'text'})

    # step 2
    # Rename the Sentiment column to rating
    all_data = all_data.rename(columns={'Sentiment': 'rating'})

    # Create a new column is_good, setting it to True if the rating is 3 or 4, and False otherwise
    all_data['is_good'] = all_data['rating'].apply(lambda x: x >= 3)
    # ## TODO: write code for steps 1 and 2 above

    # Tests to ensure your columns are labelled correctly and contain the right content
    assert 'is_good' in all_data.columns
    assert 'text' in all_data.columns
    assert 'rating' in all_data.columns
    assert sum(all_data.text.apply(lambda x: len(x))) == 868869

    #========================================================
    # make train/valid/test iterator
    #========================================================
    # Define text and label fields
    text_field = tokenize_bert
    label_field = lambda x: x

    # create dataset from dataframe 
    all_ds = DataFrameDataset(all_data, text_field, label_field)

    # Split dataset into train, valid, and test sets
    train_len, valid_len = int(0.64 * len(all_ds)), int(0.16 * len(all_ds))
    test_len = len(all_ds) - train_len - valid_len
    train_iter, valid_iter, test_iter = random_split(all_ds, [train_len, valid_len, test_len])


    #========================================================
    # make train/valid/test DataLoader
    #========================================================
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # directly use BERT vocabulary 
    vocab = tokenizer.get_vocab()
    collate_fn_with_vocab = functools.partial(collate_fn, vocab=vocab)

    BATCH_SIZE = 64
    train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_vocab)
    valid_loader = DataLoader(valid_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_vocab)
    test_loader = DataLoader(test_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_vocab)

    #========================================================
    # training RNN with BERT embeddings
    #========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Pytorch version is: ", torch.__version__)
    print("You are using: ", device)


    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    N_EPOCHS = 5

    # download the pre-trained BERT base model
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert_model = BERTwithRNN(bert,HIDDEN_DIM,OUTPUT_DIM,N_LAYERS,BIDIRECTIONAL,DROPOUT).to(device)

    # fix the bert embeddings
    for name, param in bert_model.named_parameters():                
        if name.startswith('bert'):
            param.requires_grad = False

    filename = 'Bert.pt'
    train_model(device, bert_model, train_loader, valid_loader, test_loader, filename, N_EPOCHS)

    # it will take ~17 minutes for one epoch.
    # The output will be: Epoch: 01 | Epoch Time: 17m 36s...
    # Validate accuracy is higher than 85% in the first epoch, higher than 90% in the second epoch.
