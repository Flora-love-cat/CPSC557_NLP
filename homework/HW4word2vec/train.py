from model import Word2Vec
import torch 
import torch.optim as optim
import torch.nn as nn
from data import generate_batch
import os 

def train(pairs, labels, vocabulary, 
          embed_size=100, num_epochs=50, 
          batch_size=32, learning_rate=0.01, filename='word2vec.pt'):
    vocab_size = len(vocabulary)
    model = Word2Vec(vocab_size, embed_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_pairs, batch_labels in generate_batch(pairs, labels, batch_size):
            target_words, context_words = zip(*batch_pairs)
            target_tensor = torch.tensor(target_words, dtype=torch.long)
            context_tensor = torch.tensor(context_words, dtype=torch.long)
            label_tensor = torch.tensor(batch_labels, dtype=torch.float)

            optimizer.zero_grad()
            log_sigmoid_scores = model(target_tensor, context_tensor)
            loss = criterion(log_sigmoid_scores, label_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / target_tensor.shape[0]}")

    SAVE_DIR = 'models'

    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, filename)
    if not os.path.isdir(f'{SAVE_DIR}'):
            os.makedirs(f'{SAVE_DIR}')
    
    # Save model
    model_info = {
        'state_dict': model.state_dict(),
        'vocabulary': vocabulary,
        'embed_size': model.word_embed.embedding_dim,
    }

    torch.save(model_info, MODEL_SAVE_PATH)

    return model 
