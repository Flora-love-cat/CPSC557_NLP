# ! pip install -U spacy -q
# ! python -m spacy download en_core_web_sm -q
# ! python -m spacy download de_core_news_sm -q
# ! pip install torch==1.13.1 torchtext==0.14.1 torchdata==0.5.1 -q
import locale
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import spacy
from typing import List, Iterable, Tuple 

def getpreferredencoding():
    return 'UTF-8'
locale.getpreferredencoding = getpreferredencoding

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

def tensor_transform(token_ids: List[int]):
  """
  function to add BOS/EOS and create tensor for input sequence indices
  """
  return torch.cat((torch.tensor([BOS_IDX]),
                    torch.tensor(token_ids),
                    torch.tensor([EOS_IDX])))


def collate_fn(batch: List[Tuple[str, str]], vocab_transform, token_transform, SRC, TRG, batch_first=False):
    """
    Tokenization, Numericalization, and Add BOS/EOS to create tensor for source/target language

    sort the sequences based on their lengths first, we can ensure that shorter sequences are processed together, 
    and then pad them to match the length of the longest sequence in the batch,
    reducing the number of unnecessary padding tokens that the model has to process, 
    lead to a more efficient computation and potentially better results, 

    batch: a list of tuples, where each tuple contains a pair of source and target samples. 
            The source and target samples are usually strings. e.g.
            [
              ("This is a source sentence.", "This is a target sentence."),
              ("Another source sentence.", "Another target sentence."),
              ...
            ]

    """
    src_batch, trg_batch = [], []
    
    # Sort batch based on the source length before tokenization and numericalization
    sorted_batch = sorted(batch, key=lambda x: len(x[0].rstrip("\n").split()), reverse=True)
    
    # tokenization and numericalization
    for src_sample, trg_sample in sorted_batch:
        src_batch.append(tensor_transform(vocab_transform[SRC](token_transform[SRC](src_sample.rstrip("\n")))).long())
        trg_batch.append(tensor_transform(vocab_transform[TRG](token_transform[TRG](trg_sample.rstrip("\n")))).long())
    
    # Pad sequences 
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX)

    src_len = (src_batch != PAD_IDX).sum(dim=0)

    if batch_first:
        return src_batch.T, src_len, trg_batch.T 
    else:
        return src_batch, src_len, trg_batch

def yield_tokens(data_iter: Iterable, language: str, token_transform, LANG) -> List[str]:
    for data_sample in data_iter:
        yield token_transform[language](data_sample[LANG[language]])



def token_vocab_transform(LANG, models, train_iter):
    token_transform = {}
    vocab_transform = {}

    for L, model in zip(LANG, models):
        token_transform[L] = get_tokenizer('spacy', language=model)
        vocab_transform[L] = build_vocab_from_iterator(yield_tokens(train_iter, L),
                                                            min_freq=2,
                                                            specials=special_symbols,
                                                            special_first=True)
        vocab_transform[L].set_default_index(UNK_IDX)
    
    return token_transform, vocab_transform

