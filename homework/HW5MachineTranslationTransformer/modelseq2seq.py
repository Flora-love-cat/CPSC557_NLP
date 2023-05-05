import torch 
import torch.nn as nn 
import random 


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, enc_dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, num_layers=num_layers, bidirectional = True)
        self.fc = nn.Linear(2 * enc_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(enc_dropout)
        
    def forward(self, src, src_len):
        """
        a stacked bidirectional RNN encoder, receive a source sequence, produce a context vector for decoder

        input 
        src: [src sent len, batch size]
        src_len: [src sent len]

        return
        outputs: [sent len, batch size, enc hid dim * 2]
        hidden: [batch size, dec_hid_dim]
        """
        embedded = self.dropout(self.embedding(src)) # [src sent len, batch size, emb dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu())
        # hidden = [n layers * num directions, batch size, dec_hid_dim]
        # hidden is stacked, first dimension is [forward^1_T, backward^1_T, forward^2_T, backward^2_T, ...]
        packed_outputs, hidden = self.rnn(packed_embedded)
        # outputs = [sent len, batch size, dec_hid_dim * num directions]
        # outputs are always from the last layer , first dimension is [h^L_1, h^L_2, ..., h^L_T]
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
    
        # initial decoder hidden is concatenation of final hidden state of the forwards `hidden [-2, :, : ]` and backwards encoder RNNs `hidden [-1, :, : ]` fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        
        return outputs, hidden
    


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dec_dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers)
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dec_dropout)

    def forward(self, input, hidden):
        """
        stacked unidirectional RNN decoder, produce target token one by one 
        input:
        input: [batch size]
        hidden: [n layers, batch size, hid dim]. first hidden is context 
        cell (if LSTM): [n layers, batch size, hid dim]

        prediction: [batch size, output dim]
        hidden: [n layers, batch size, hid dim]
        cell (if LSTM): [n layers, batch size, hid dim]
        """
        input = input.unsqueeze(0)  # [batch_size] -> [1, batch_size] sentence length is 1
        embedded = self.dropout(self.embedding(input)) # [1, batch_size, emb_dim]
        output, hidden = self.rnn(embedded, hidden) # output: [1, batch size, hid dim] sentence length is 1
        prediction = self.out(output.squeeze(0))
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.enc_hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        """
        src = [src sent len, batch size]
        trg = [trg sent len, batch size]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        """
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        _, hidden = self.encoder(src, src_len)
        hidden = hidden.unsqueeze(0)
        # first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):  
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            # choose whether to do teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            _, top1 = output.max(dim=1)   # (val_max, idx_max)
            input = (trg[t] if teacher_force else top1)
        
        return outputs
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, hidden, encoder_outputs, mask):
        """
        hidden: [batch size, dec hid dim]
        encoder_outputs: hidden states (bidirectional) over the entire src sentence
                        [src sent len, batch size, enc hid dim * 2] 
        mask: [batch size, src sent len]
        """
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        # TODO 1: repeat encoder hidden state src_len times
        # hidden = [batch size, src sent len, dec hid dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # TODO 2: swap dimension src sent len, batch size of encoder outputs for matrix multiplication
        #  (src sent len, batch size, enc hid dim x 2)-> (batch size, src sent len, enc hid dim x 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # compute energy [batch size, src sent len, dec hid dim]
        # swap dimension dec hid dim, src sent len of energy for matrix multiplication [batch size, dec hid dim, src sent len]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))).permute(0, 2, 1)
        
        # repeat v to allow for matrix multiplication. [dec hid dim] -> [batch size, 1, dec hid dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)         
        
        # TODO 4: compute attention score and mask the padded input indices. attention = [batch size, src sent len]
        attention = torch.bmm(v, energy).squeeze(1).masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim=1)




class DecoderAtten(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers)
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, input, hidden, encoder_outputs, mask):
        """
        input: [batch size]
        hidden: [batch size, dec hid dim]
        encoder_outputs: [src sent len, batch size, enc hid dim * 2]
        mask: [batch size, src sent len]
        """
        input = input.unsqueeze(0) # input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input)) # embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs, mask).unsqueeze(1) # a = [batch size, 1, src sent len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        
        # TODO 1: batch matrix multiplication [batch size, 1, enc hid dim * 2] -> [1, batch size, enc hid dim * 2]
        weighted = torch.bmm(a, encoder_outputs).permute(1, 0, 2)
        
        # TODO 2: [1, batch size, (enc hid dim * 2) + emb dim]
        rnn_input = torch.cat((embedded, weighted), dim = 2) 
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]
        
        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()
        # print(output.shape, hidden.shape, output[0,0,:25], hidden[0,0,:25])
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        output = self.out(torch.cat((output, weighted, embedded), dim=1)) # output = [batch size, output dim]
        
        return output, hidden.squeeze(0), a.squeeze(1)
    

class Seq2SeqAtten(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, sos_idx, eos_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device
        
    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        """
        src = [src sent len, batch size]
        src_len = [batch size]
        trg = [trg sent len, batch size]
        teacher_forcing_ratio is probability to use teacher forcing
            e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        """
        if trg is None:
            inference = True
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            trg = torch.zeros((100, src.shape[1]), dtype=torch.long).fill_(self.sos_idx).to(src.device)
        else:
            inference = False
            
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        # tensor to store attention
        attentions = torch.zeros(max_len, batch_size, src.shape[0]).to(self.device)
        
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)
                
        # first input to the decoder is the <sos> tokens
        output = trg[0,:]
        
        mask = self.create_mask(src)   # mask = [batch size, src sent len]
                
        for t in range(1, max_len):
            output, hidden, attention = self.decoder(output, hidden, encoder_outputs, mask)
            outputs[t] = output
            attentions[t] = attention
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)
            if inference and output.item() == self.eos_idx:
                return outputs[:t], attentions[:t]
            
        return outputs, attentions