"""
POS tagging for Wiki Spanish corpus 
"""
import sys
import time
import HMMPOS


START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000

DATA_PATH = './data/'
OUTPUT_PATH = './output/'

def main():
    time.perf_counter()

    infile = open(DATA_PATH + "wikicorpus_tagged_train.txt", "r")

    train = infile.readlines()
    infile.close()

    words, tags = HMMPOS.split_wordtags(train)

    if len(sys.argv) > 1 and sys.argv[1] == "-reverse":
        q_values = HMMPOS.calc_trigrams_reverse(tags)
    else:
        q_values = HMMPOS.calc_trigrams(tags)

    HMMPOS.q2_output(q_values, OUTPUT_PATH + 'C2.txt')

    known_words = HMMPOS.calc_known(words)

    words_rare = HMMPOS.replace_rare(words, known_words)

    HMMPOS.q3_output(words_rare, OUTPUT_PATH + "C3.txt")

    e_values, taglist = HMMPOS.calc_emission(words_rare, tags)

    HMMPOS.q4_output(e_values, OUTPUT_PATH + "C4.txt")

    del train
    del words_rare

    infile = open(DATA_PATH + "wikicorpus_dev.txt", "r")

    dev = infile.readlines()
    infile.close()

    dev_words = [sentence.split() for sentence in dev]

    viterbi_tagged = HMMPOS.viterbi(dev_words, taglist, known_words, q_values, e_values)

    HMMPOS.q5_output(viterbi_tagged, OUTPUT_PATH + 'C5.txt')

    nltk_tagged = HMMPOS.nltk_tagger(words, tags, dev_words)

    HMMPOS.q6_output(nltk_tagged, OUTPUT_PATH + 'C6.txt')

    print("Part C time: " + str(time.clock()) + ' sec')

if __name__ == "__main__": main()
