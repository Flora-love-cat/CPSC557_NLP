"""
POS tagging by HMM Viterbi algorithm or NLTK back-off tagger
"""

import sys
import nltk
import math
import time
from typing import List, Tuple 
from collections import Counter, defaultdict
import pprint
pp = pprint.PrettyPrinter(indent=4)

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: Problem 1 IMPLEMENT THIS FUNCTION
def split_wordtags(brown_train: List[str]) -> Tuple[List, List]:
    """
    Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
    
    brown_train: Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
    
    return:
    brown_words (the list of words): every element is a list of the tags of a particular sentence.
    brown_tags (the list of tags): every element is a list of the tags of a particular sentence.
    """
    brown_words = []
    brown_tags = []

    for sentence in brown_train:
        # prepare two separate lists
        words, tags = [START_SYMBOL] * 2, [START_SYMBOL] * 2

        split_sentence = sentence.split() # split the sentence by whitespace

        # consider each token
        for token in split_sentence:
            word, tag = token.rsplit("/", 1) # split by "/"
            words.append(word)
            tags.append(tag)
            
        # add end symbols
        words.append(STOP_SYMBOL)
        tags.append(STOP_SYMBOL)

        # append the two lists
        brown_words.append(words)
        brown_tags.append(tags)

    return brown_words, brown_tags

# TODO: Problem 2 IMPLEMENT THIS FUNCTION
def calc_trigrams(brown_tags: List[str]) -> dict:
    """
    takes tags from the training data and calculates tag trigram probabilities.
    
    brown_tags (the list of tags): every element is a list of the tags of a particular sentence.
    returns 
    q_values: trigram probabilities. a dictionary where the keys are tuples that represent the tag trigram, 
                and the values are the log probability of that trigram
    """
    # tokenize sentence to bigram and trigram, respectively

    bigram_tokenized_sentences = [bigram for split_sentence in brown_tags for bigram in list(nltk.bigrams(split_sentence))]
    trigram_tokenized_sentences = [trigram for split_sentence in brown_tags for trigram in list(nltk.trigrams(split_sentence)) ]
   
    # count bigram and trigram, respectively
    bigram_counter = Counter(bigram_tokenized_sentences) 
    trigram_counter = Counter(trigram_tokenized_sentences) 

    # dictionary of log trigram probabilities
    q_values = defaultdict(float, {trigram: math.log(tri_count / bigram_counter[(trigram[0], trigram[1])], 2) for trigram, tri_count in trigram_counter.items()})
    
    return q_values

#TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates the tag trigrams in reverse.  
# In other words, instead of looking at the probabilities that the third tag follows the first two, 
# look at the probabilities of the first tag given the next two.
# Hint: This code should only differ slightly from calc_trigrams(brown_tags)
def calc_trigrams_reverse(brown_tags):
    q_values = {}
    # have a list of bigrams
    bigram_tokenized_sentences = []
    # have a list of trigrams
    trigram_tokenized_sentences = []
    # iterate over the sentences
    for split_sentence in brown_tags:
        # create a list of trigrams and append to total list
        trigram_tokenized_sentences.extend(list(nltk.trigrams(split_sentence[::-1])))
        # create a list of bigrams and append to total list
        bigram_tokenized_sentences.extend(list(nltk.bigrams(split_sentence[::-1])))
    # count up the bigrams
    bigram_counter = Counter(bigram_tokenized_sentences)
    # count up the trigrams
    trigram_counter = Counter(trigram_tokenized_sentences)
    # sum up all the trigrams
    trigram_sum = sum(trigram_counter.itervalues())
    # compute the trigram probabilities
    for trigram in trigram_counter:
        tri_count = trigram_counter[trigram]
        bi_count = bigram_counter[(trigram[0], trigram[1])]
        q_values[trigram] = math.log(tri_count / bi_count, 2)
    # return the dictionary of log probabilities
    return q_values


def q2_output(q_values: dict, filename: str):
    """takes output from calc_trigrams() and outputs it in the proper format
    """    
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    for trigram in sorted(trigrams):
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: Problem 3 IMPLEMENT THIS FUNCTION
def calc_known(brown_words: List[List]) -> set:
    """
    Takes the words from the training data and returns a set of frequent words
    
    brown_words: a list where every element is a list of the words of a particular sentence.

    return:
    known_words: a set of all of the words that occur more than 5 times 
    """
    word_list = [word for sentence in brown_words for word in sentence]
    word_counter = Counter(word_list)
    known_words = set([word for word, count in word_counter.items() if count > RARE_WORD_MAX_FREQ])

    return known_words

# TODO: Problem 3 IMPLEMENT THIS FUNCTION
def replace_rare(brown_words: List[List], known_words: set) -> List[List]:
    """
    Replace rare words in training data
    input:
    brown_words: training data
    known_words: a set of words that should not be replaced
    
    Returns 
    brown_words_rare: equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
    """
    brown_words_rare = [[token if token in known_words else RARE_SYMBOL for token in sentence] for sentence in brown_words]

    return brown_words_rare


def q3_output(rare: List[List], filename: str):
    # This function takes the ouput from replace_rare and outputs it to a file
    outfile = open(filename, 'w')
    for sentence in rare:
        # skip the first 2 start symbols in each sentence
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: problem 4 IMPLEMENT THIS FUNCTION
def calc_emission(brown_words_rare: List[List], brown_tags: List[str]) -> Tuple[dict, set]:
    """
    Calculates emission probabilities and creates a set of all possible tags
    
    return
    e_values: emission probabilities. a dict where each key is a tuple of a word and a tag, 
                and value is log probability of emission of the word given the tag
    taglist: a set of all possible tags for this data set
    """
    # a list of (word, tag) tuples, a list of tags
    word_tags, tags = [], []
    for word_sentence, tag_sentence in zip(brown_words_rare, brown_tags):
        word_tags.extend(zip(word_sentence, tag_sentence))
        tags.extend(tag_sentence)

    word_tag_counter = Counter(word_tags) # count up all the (word, tag) pairs
    tag_counter = Counter(tags)  # count up all the tags

    # the probability of an emission is the Count((word, tag)) / Count(tag)
    e_values = {word_tag : math.log(word_tag_counter[word_tag] / tag_counter[word_tag[1]], 2) for word_tag in word_tag_counter}

    taglist = set([token for sentence in brown_tags for token in sentence])

    return e_values, taglist


def q4_output(e_values, filename):
    # This function takes the output from calc_emissions() and outputs it
    outfile = open(filename, "w")
    emissions = e_values.keys()
    for item in sorted(emissions):
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
def viterbi(brown_dev_words: List[List[str]], taglist: set, known_words: set, q_values: dict, e_values: dict) -> List[str]:
    """
    Viterbi algorithm for HMM tagger.
    
    params:
    brown_dev_words: development data. a list where every element is a list of the words of a particular sentence.
    taglist: a set of all possible tags
    known_words: a set of all known words
    q_values: trigram probabilities. output of calc_trigrams(). a dict where the keys are tuples that represent the tag trigram, 
                and the values are the log probability of that trigrama 
    e_values: emission probabilities. output of calc_emissions(). a dict where each key is a tuple of a word and a tag, 
                and value is log probability of emission of the word given the tag

    return: 
    tagged: a list of tagged sentences in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data
            do not contain the "_RARE_" symbol, but rather the original words of the sentence.
    """
    # a list of tagged sentences
    tagged = []

    for split_sentence in brown_dev_words:
        ############ INITIALIZATION
        pi = defaultdict(float) # pi[(k, u, v)]: max prob of tag sequence u, v at position k
        bp = {} # bp[(k, u, v)]: backpointers i.e. the argmax of pi[(k, u, v)]
        pi[(0, START_SYMBOL, START_SYMBOL)] = 1.0 # the base case
        split_sentence_refine = [word if word in known_words else RARE_SYMBOL for word in split_sentence]
        n = len(split_sentence_refine)
        ############ VITERBI
        for k in range(1, n + 1):
            index = k - 1
            ######### STORE WORDS
            v_word = split_sentence_refine[index]
            v_tags = [tag for tag in taglist if (v_word, tag) in e_values]
            u_tags = [START_SYMBOL]
            if index > 0:
                u_word = split_sentence_refine[index - 1]
                u_tags = [tag for tag in taglist if (u_word, tag) in e_values]
            w_tags = [START_SYMBOL]
            if index > 1:
                w_word = split_sentence_refine[index - 2]
                w_tags = [tag for tag in taglist if (w_word, tag) in e_values]
            ######### BEGIN ITERATION
            # iterate over the tags available
            for u_tag in u_tags:
                for v_tag in v_tags:
                    best_prob = float('-Inf')
                    best_tag = None
                    for w_tag in w_tags:
                        if (v_word, v_tag) in e_values:
                            total_prob = (
                                pi.get((k - 1, w_tag, u_tag), LOG_PROB_OF_ZERO) +
                                q_values.get((w_tag, u_tag, v_tag), LOG_PROB_OF_ZERO) +
                                e_values.get((v_word, v_tag)) # (THE, DET) == -4.23
                            )
                            if total_prob > best_prob:
                                best_prob = total_prob
                                best_tag = w_tag
                    pi[(k, u_tag, v_tag)] = best_prob
                    bp[(k, u_tag, v_tag)] = best_tag
        ######### DO THE SAME FOR STOP
        best_prob = float('-Inf')
        best_u_tag = None
        best_v_tag = None
        # collect tags
        u_word = split_sentence_refine[n - 2]
        u_tags = [tag for tag in taglist if (u_word, tag) in e_values]
        v_word = split_sentence_refine[n - 1]
        v_tags = [tag for tag in taglist if (v_word, tag) in e_values]
        # find best probs
        for u_tag in u_tags:
            for v_tag in v_tags:
                total_prob = (
                        pi.get((n, u_tag, v_tag), LOG_PROB_OF_ZERO) +
                        q_values.get((u_tag, v_tag, STOP_SYMBOL), LOG_PROB_OF_ZERO)
                    )
                if total_prob > best_prob:
                    best_prob = total_prob
                    best_u_tag = u_tag
                    best_v_tag = v_tag
        ######## RECOVER SENTENCE FROM BACKPOINTERS
        tagged_sentence = [best_v_tag, best_u_tag]
        # build up the tags
        for i, k in enumerate(range(n - 2, 0, -1)):
            tagged_sentence.append(bp[(k + 2, tagged_sentence[i + 1], tagged_sentence[i])])
        tagged_sentence.reverse()
        # build up the full sentence
        full_sentence = []
        for word_index in range(0, n):
            full_sentence.append(split_sentence[word_index] + '/' + tagged_sentence[word_index])
        tagged.append(' '.join(full_sentence)+'\n')

    return tagged


def q5_output(tagged: List[str], filename: str):
    # This function takes the output of viterbi() and outputs it to file
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: Problem 6. IMPLEMENT THIS FUNCTION
def nltk_tagger(brown_words: List[List], brown_tags: List[List], brown_dev_words: List[List]) -> List[str]:
    """
    create an instance of NLTK’s trigram tagger set to back off to NLTK’s bigram tagger.
    Let the bigram tagger itself back off to NLTK’s default tagger using the tag “NOUN”.
    
    brown_words and brown_tags: training data and labels
    brown_dev_words: data that should be tagged

    return:
    tagged: a list of tagged sentences in the format "WORD/TAG", separated by spaces. 
            Each sentence is a string with a terminal newline, not a list of tokens.
    """
    # make training set for NLTK tagger [[(token, tag), ..., (token, tag)], ..., []]
    train_sentences = [list(zip(tokens, tags)) for tokens, tags in zip(brown_words, brown_tags)]
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(train_sentences, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(train_sentences, backoff=bigram_tagger)
    tagged = []
    # use the taggers to tag the sentence
    for split_sentence in brown_dev_words:
        tagged_sentence = [word + '/' + tag for word, tag in trigram_tagger.tag(split_sentence)]
        tagged.append(' '.join(tagged_sentence) + '\n')

    return tagged


def q6_output(tagged, filename):
    # This function takes the output of nltk_tagger() and outputs it to file
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = './data/'
OUTPUT_PATH = './output/'

def main():
    # start timer
    time.perf_counter()

    # load Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # tokenize training data and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities with an option to reverse (question 7)
    if len(sys.argv) > 1 and sys.argv[1] == "-reverse":
        q_values = calc_trigrams_reverse(brown_tags)
    else:
        q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    ########## question 3
    # calculate list of words with count > 5
    known_words = calc_known(brown_words)
    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)
    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    ########### question 4
    # calculate emission probabilities
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)
    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare


    # load Brown development data
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # tokenize Brown development data 
    brown_dev_words = [sentence.split() for sentence in brown_dev]

    ######## question 5 POS tagging by HMM Viterbi algorithm
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    ####### question 6 POS tagging by NLTK back-off tagger
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print("Part B time: " + str(time.perf_counter()) + ' sec')

if __name__ == "__main__": main()
