# ! pip install -U spacy -q
# ! python -m spacy download en_core_web_md -q
# ! pip install torchtext 
import spacy
import en_core_web_md
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from typing import List, Tuple 
from collections import Counter 
from transformers import BertTokenizer

def tokenize_spacy(text):
    #load Spacy English median model
    text_to_nlp = en_core_web_md.load(disable=["parser", "tagger", "ner"])
    return [token.text for token in text_to_nlp(text)]


def tokenize_bert(text):
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

    tokens = tokenizer.tokenize(text)[:max_input_length - 2]
    return ['[CLS]'] + tokens + ['[SEP]']


class DataFrameDataset(Dataset):
    """Convert pandas DataFrame to torch Dataset"""
    def __init__(self, df, text_field, label_field):
        self.df = df
        self.text_field = text_field
        self.label_field = label_field

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx] 
        text = self.text_field(row['text'])
        label = row['is_good']
        return text, label


def build_vocab(iterator, max_size=25_000):
    counter = Counter()
    for tokens, labels in iterator:
        counter.update(tokens)
    vocab = Vocab(counter.most_common(max_size))

    # Define special symbols and indices
    UNK_IDX, PAD_IDX = 0, 1
    vocab = {token: i+2 for i, (token, count) in enumerate(vocab.vocab)}
    vocab['[UNK]'] = UNK_IDX
    vocab['[PAD]'] = PAD_IDX

    return vocab 




def collate_fn(batch: List[Tuple[str, str]], vocab):
    """
    Tokenization, Numericalization, and Add unknown / pad token to create tensor for texts and labels

    sort the sequences based on their lengths first, we can ensure that shorter sequences are processed together, 
    and then pad them to match the length of the longest sequence in the batch,
    reducing the number of unnecessary padding tokens that the model has to process, 
    lead to a more efficient computation and potentially better results, 

    batch: a list of tuples, where each tuple contains a pair of text (string) and label (bool). 
            [
              ("This is a source sentence.", True),
              ("Another source sentence.", False),
              ...
            ]
    """
    texts, labels = zip(*batch)
   
    # Sort texts and labels based on the length of texts in descending order
    sorted_indices = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    texts = [texts[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    
    # Convert texts to indices using the vocabulary
    texts = [[vocab[token] if token in vocab else vocab['[UNK]'] for token in text] for text in texts]
    
    # Convert texts and labels to tensors
    texts = [torch.tensor(text, dtype=torch.long) for text in texts]
    labels = torch.tensor(labels, dtype=torch.float)
    
    # Pad sequences
    texts = pad_sequence(texts, padding_value=vocab['[PAD]'], batch_first=True)
    
    return texts, labels

