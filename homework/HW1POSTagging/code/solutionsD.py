"""
Neural POS tagging
"""
import neuralPOStagger as neural_tagger

import numpy as np

PAD = "__PAD__"
UNK = "__UNK__"
DIM_EMBEDDING = 100
LSTM_HIDDEN = 100
BATCH_SIZE = 20
LEARNING_RATE = 0.015
LEARNING_DECAY_RATE = 0.05
EPOCHS = 10
KEEP_PROB = 0.5
GLOVE = "./data/glove.6B.100d.txt"
WEIGHT_DECAY = 1e-8

import torch
torch.manual_seed(0)


def main():
   # parser = argparse.ArgumentParser(description='POS tagger.')
   # parser.add_argument('training_data')
   # parser.add_argument('dev_data')
   # args = parser.parse_args()

    train = neural_tagger.read_data("data/Brown_tagged_train.txt")
    dev = neural_tagger.read_data("data/Brown_tagged_dev.txt")

    # Make indices
    id_to_token = [PAD, UNK]
    token_to_id = {PAD: 0, UNK: 1}
    id_to_tag = [PAD]
    tag_to_id = {PAD: 0}

    for tokens, tags in train + dev:
        for token in tokens:
            token = neural_tagger.simplify_token(token)
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
                id_to_token.append(token)
        for tag in tags:
            if tag not in tag_to_id:
                tag_to_id[tag] = len(tag_to_id)
                id_to_tag.append(tag)

    NTAGS = len(tag_to_id)

    # Load pre-trained GloVe vectors
    pretrained = {}
    for line in open(GLOVE):
        parts = line.strip().split()
        word = parts[0]
        vector = [float(v) for v in parts[1:]]
        pretrained[word] = vector
    pretrained_list = []
    scale = np.sqrt(3.0 / DIM_EMBEDDING)
    for word in id_to_token:
        # apply lower() because all GloVe vectors are for lowercase words
        if word.lower() in pretrained:
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            random_vector = np.random.uniform(-scale, scale, [DIM_EMBEDDING])
            pretrained_list.append(random_vector)
    pretrained_embed = np.array(pretrained_list)

    # Model creation
    model = neural_tagger.TaggerModel(NTAGS, pretrained_embed)
    # Create optimizer and configure the learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY)

    expressions = (model, optimizer)

    # Load model
    model.load_state_dict(torch.load('./output/tagger.pt'))

    # Evaluation pass.
    _, test_acc = neural_tagger.do_pass(dev, token_to_id, tag_to_id, expressions, False)
    print("Test Accuracy: {:.3f}".format(test_acc*100))

if __name__ == '__main__':
    main()
