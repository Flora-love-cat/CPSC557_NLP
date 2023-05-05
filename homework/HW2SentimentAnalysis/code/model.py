import torch.nn as nn
import torch

class SRN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        
    def forward(self, text):
        """
        In simple RNNs, there isn't a significant difference between using the last output and the final hidden state. 
        However, for LSTM and GRU networks, the final hidden state can contain more abstract and high-level representations of the input sequence, 
        making it more useful for tasks like sentiment analysis.
        
        input: 
        text: (batch_size, seq_len)

        return:
        logits (batch_size)
        """
        embed = self.embedding(text)  # (batch_size, seq_len, hidden_dim)
        output, hidden = self.rnn(embed)  # output (batch_size, seq_len, num_directions * hidden_dim), hidden [num_layers * num_directions, batch_size, hidden_dim]
        logits = self.fc(hidden.squeeze(0)).squeeze(1)    # hidden [batch_size, hidden_dim]  

        return logits  # (batch_size)
    
class LSTM(nn.Module):
    # TODO: IMPLEMENT THIS FUNCTION
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, bidirectional=True):
        """    
        Initialize the three layers in the RNN, self.embedding, self.rnn, and self.fc
        Each one has a corresponding function in nn
        embedding maps from input_dim->embedding_dim
        rnn maps from embedding_dim->hidden_dim
        fc maps from hidden_dim*2->output_dim
        """
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        
    def forward(self, text):
        """
        x has dimensions [sentence length, batch size]
        embedded has dimensions [sentence length, batch size, embedding_dim]
        output has dimensions [sentence length, batch size, hidden_dim*2] (since bidirectional)
        hidden has dimensions [2, batch size, hidden_dim]
        cell has dimensions [2, batch_size, hidden_dim]

        return: 
        logits (batch_size)
        """
        embed = self.embedding(text)
        output, (hidden, cell) = self.rnn(embed)
        # Concatenate the final forward and backward hidden layers
        if self.rnn.bidirectional:  
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) #hidden = [batch size, hidden dim*2]
        else:
            hidden = hidden[-1,:,:] #hidden = [batch size, hidden dim]

        logits = self.out(hidden).squeeze(1)  # [64, 1] => [64]
        return logits  


class BERTwithRNN(nn.Module):
    def __init__(self,bert,hidden_dim,output_dim,n_layers,bidirectional,dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        """
        input: 
        text: [batch size, sent len]

        output:
        logits: [batch size]
        """
        
        with torch.no_grad():
            embedded = self.bert(text)[0] #embedded = [batch size, sent len, emb dim]
        _, hidden = self.rnn(embedded) #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)) #hidden = [batch size, hidden dim*2]
        else:
            hidden = self.dropout(hidden[-1,:,:]) #hidden = [batch size, hidden dim]
        
        logits = self.out(hidden).squeeze(1) #output = [batch size, out dim]
        
        return logits