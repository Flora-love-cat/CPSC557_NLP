
import pandas as pd
from data import preprocess_sentence, create_vocabulary, generate_training_data, generate_negative_samples, 
from train import train 
import torch 
from model import Word2Vec



# Finnish and Russian parallel corpora

finrus = pd.read_csv('/finrus.csv', index_col=0)
# English and Russian parallel corpora
engrus = pd.read_csv('/engrus.csv', index_col=0)
# English and Finnish parallel corpora
engfin = pd.read_csv('/engfin.csv', index_col=0)

pp_eng2fin = [preprocess_sentence(sentence, 'english') for sentence in engfin['english']]
pp_eng2rus = [preprocess_sentence(sentence, 'english') for sentence in engrus['english']]
pp_fin2eng = [preprocess_sentence(sentence, 'finnish') for sentence in engfin['finnish']]
pp_fin2rus = [preprocess_sentence(sentence, 'finnish') for sentence in finrus['finnish']]
pp_rus2eng = [preprocess_sentence(sentence, 'russian') for sentence in engrus['russian']]
pp_rus2fin = [preprocess_sentence(sentence, 'russian') for sentence in finrus['russian']]


vocabulary_english, token_counts_english = create_vocabulary(pp_eng2fin, pp_eng2rus)
print('english vocabulary length:',len(vocabulary_english))

vocabulary_finnish, token_counts_finnish = create_vocabulary(pp_fin2eng, pp_fin2rus)
print('english vocabulary length:',len(vocabulary_finnish))

vocabulary_russian, token_counts_russian = create_vocabulary(pp_rus2eng, pp_rus2fin)
print('russian vocabulary length:',len(vocabulary_russian))


window_size = 2
m = 2 # number of negative samples per positive sample
x_english, y_english = generate_training_data(pp_eng2fin, pp_eng2rus, vocabulary_english, window_size)
x_finnish, y_finnish = generate_training_data(pp_fin2eng, pp_fin2rus, vocabulary_finnish, window_size)
x_russian, y_russian = generate_training_data(pp_rus2eng, pp_rus2fin, vocabulary_russian, window_size)

pairs_english, labels_english = generate_negative_samples(x_english, y_english, m, vocabulary_english, token_counts_english)
pairs_finnish, labels_finnish = generate_negative_samples(x_finnish, y_finnish, m, vocabulary_finnish, token_counts_finnish)
pairs_russian, labels_russian = generate_negative_samples(x_russian, y_russian, m, vocabulary_russian, token_counts_russian)


batch_size = 32
embed_size = 100
learning_rate = 0.001
num_epochs = 500



# Train the models for each language
word2vec_english = train(pairs_english, labels_english, vocabulary_english, filename='word2vecEnglish.pt')
word2vec_finnish = train(pairs_finnish, labels_finnish, vocabulary_finnish, filename='word2vecFinnish.pt')
word2vec_russian = train(pairs_russian, labels_russian, vocabulary_russian, filename='word2vecRussian.pt')

# English word embedding lookup table 
wordeng_embed = word2vec_english.word_embed.weight.detach().numpy()


# Load model
model_info = torch.load('word2vecEnglish.pt')
loaded_model = Word2Vec(len(model_info['vocabulary']), model_info['embed_size'])
loaded_model.load_state_dict(model_info['state_dict'])
loaded_vocabulary = model_info['vocabulary']