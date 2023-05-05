import torch 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker


def translate_sentence(model, sentence, token_transform, vocab_transform, SRC, TRG):
    tokenized = token_transform[SRC](sentence) # tokenize sentence
    
    tokenized = ['<bos>'] + [t.lower() for t in tokenized] + ['<eos>'] # add <bos> and <eos> tokens and lowercase
    
    numericalized = [vocab_transform[SRC][t] for t in tokenized] # convert tokens into indexes
    
    sentence_length = torch.LongTensor([len(numericalized)]) # need sentence length for masking
    
    tensor = torch.LongTensor(numericalized).unsqueeze(1) # convert to tensor and add batch dimension
    
    translation_tensor_probs, attention = model(tensor, sentence_length, None, 0) # pass through model to get translation probabilities
    
    translation_tensor = torch.argmax(translation_tensor_probs.squeeze(1), 1) # get translation from highest probabilities
    
    translation = vocab_transform[TRG].lookup_tokens(list(translation_tensor.cpu().numpy()))[1:] # ignore the first token, just like we do in the training loop
    
    return translation, attention[1:] # ignore first attention array



def display_attention(source, translation, attention, token_transform, SRC):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    attention = attention[:len(translation)].squeeze(1).cpu().detach().numpy() # cut attention to same length as translation
    cax = ax.matshow(attention, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + ['<bos>'] + [t.lower() for t in token_transform[SRC](source)] + ['<eos>'], rotation=90)
    ax.set_yticklabels([''] + translation)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


