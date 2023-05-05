
# nltk.download('punkt')
# nltk.download('stopwords')
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict, Tuple 
from collections import Counter, OrderedDict
import random 
import numpy as np 

stop_words_eng = set(stopwords.words('english')) # 179 tokens
stop_words_fin = set(stopwords.words('finnish')) # 229 tokens
stop_words_rus = set(stopwords.words('russian')) # 151 tokens


def preprocess_sentence(text: str, language: str) -> List[str]:
    """
    tokenization

    return
    words: a list of tokens
    """
    # Tokenize
    words = word_tokenize(text)
    
    # Lowercase
    words = [word.lower() for word in words]
    
    # Remove punctuation
    words = [word for word in words if word not in string.punctuation]
    
    # Remove stopwords
    if language == 'english':
        words = [word for word in words if word not in stop_words_eng]
    elif language == 'finnish':
        words = [word for word in words if word not in stop_words_fin]
    elif language == 'russian':
        words = [word for word in words if word not in stop_words_rus]
    else:
        raise ValueError("Unsupported language")
    
    return words


def create_vocabulary(corpus1: List[List], corpus2: List[List]) -> Tuple[Dict, Dict]:
    """
    params:
    corpus1, corpus2: corpus from two different languages. a list of list of tokens

    return
    vocabulary: a dict of token to index
    token_counts: a dict of token to count
    """
    all_words = [word for word_list in corpus1+corpus2 for word in word_list]
    token_counts = dict(Counter(all_words))
    # Create an OrderedDict from the sorted token counts
    ordered_token_counts = OrderedDict(sorted(token_counts.items(), key=lambda x: -x[1]))
    vocabulary = {word: i for i, word in enumerate(ordered_token_counts)}

    return vocabulary, ordered_token_counts
 


def generate_training_data(corpus1, corpus2, vocabulary, window_size):
    x = []
    y = []

    for sentence in (corpus1 + corpus2):
        for target_index, target_word in enumerate(sentence):
            # Ensure the target word is in the vocabulary
            if target_word not in vocabulary:
                continue

            # Get the context words within the window
            context_start = max(0, target_index - window_size)
            context_end = min(len(sentence), target_index + window_size + 1)

            for context_index, context_word in enumerate(sentence[context_start:context_end]):
                # Skip the target word itself and Ensure the context word is in the vocabulary
                if (context_index + context_start == target_index) or context_word not in vocabulary:
                    continue
                x.append(vocabulary[target_word])
                y.append(vocabulary[context_word])

    return x, y


def generate_negative_samples(x: List[int], y: List[int], m: int, vocabulary: Dict, token_counts: Dict) -> Tuple[List[int], List[int]]:
    """
    x, y: a list of indices of target words, a list of indices of context words
    m: number of negative pairs per positive pairs
    vocabulary: a dict of token to index
    token_counts: a dict of token to count

    return:
    pairs: a list of shuffled positive pairs and m times negative paris
    labels: a list of binary labels (0: negative, 1: positive)
    """
    pairs = []
    labels = []

    words, frequencies = zip(*token_counts.items())
    # Calculate the probability of each word based on its frequency
    probabilities = [freq / sum(frequencies) for freq in frequencies]

    for target_word_index, context_word_index in zip(x, y):
        # Add the positive (true) pair
        pairs.append((target_word_index, context_word_index))
        labels.append(1)

        # Add m negative (false) pairs
        for _ in range(m):
            random_word = np.random.choice(words, p=probabilities)
            while (vocabulary[random_word] == target_word_index) or (vocabulary[random_word] == context_word_index):
                random_word = np.random.choice(words, p=probabilities)

            pairs.append((target_word_index, vocabulary[random_word]))
            labels.append(0)

    # Shuffle the pairs and labels in the same order
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined)

    return pairs, labels 


def generate_batch(pairs, labels, batch_size):
    indices = np.arange(len(pairs))
    np.random.shuffle(indices)

    for i in range(0, len(pairs), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_pairs = [pairs[index] for index in batch_indices]
        batch_labels = [labels[index] for index in batch_indices]
        yield batch_pairs, batch_labels