import torch.nn as nn 
import torch 
import torch.nn.functional as F 


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        # Weight matrices for query, key, and value
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        
        # Final fully connected layer to project the concatenated attention matrix
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)

        # self.scale is not a parameter or buffer of the model, but a constant tensor
        # it won't be automatically transferred to the device when you call `model.to(device)`
        # so we need to attach is to device in model definition
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        
    def forward(self, query, key, value, mask=None):
        """
        query = key = value: [batch size, sent len, hid dim] 
        """
        batch_size = query.shape[0]
        # Compute Q, K, V using linear layers. 
        # Q, K, V = [batch size, sent len, hid dim]  
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # TODO 1: Reshape and permute Q, K, V.
        # Q, K, V = [batch size, sent len, n_heads, hid dim // n heads] -> (batch size, n_heads, sent len, hid dim // n heads)
        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # Q, K, V = [batch size, n heads, sent len, hid dim // n heads]
        
        # TODO 2: compute attention scores (energy) by dot product of Q and K and scaling
        # energy = [batch size, n heads, sent len, sent len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        # apply masked self-attention if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # Compute the softmax-normalized attention scores. 
        # attention = [batch size, n heads, sent len, sent len]
        attention = self.do(F.softmax(energy, dim=-1))

        # TODO 3: compute attention matrix for each head by multiplying the attention scores with V
        # x = [batch size, n heads, sent len, hid dim // n heads] -> [batch size, sent len, n heads, hid dim // n heads]
        x = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        
        # TODO 4 concatenate the attention matrices for all heads. 
        # x = [batch size,sent len, hid dim]
        x = x.view(batch_size, -1, self.hid_dim)
        
        # Project the concatenated attention matrix to the output shape. 
        # x = [batch size, sent len, hid dim]
        x = self.fc(x)
        
        return x
    

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        
        # Two convolution layers (1x1)
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        
        self.do = nn.Dropout(dropout)
        
    def forward(self, x):
        # x = [batch size, sent len, hid dim] -> [batch size, hid dim, sent len]
        x = self.do(F.relu(self.fc_1(x.permute(0, 2, 1)))) # x = [batch size, ff dim, sent len]
        
        x = self.fc_2(x).permute(0, 2, 1) # x = [batch size, hid dim, sent len] -> [batch size, sent len, hid dim]
        
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()
        
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        """
        src = [batch size, src sent len, hid dim]
        src_mask = [batch size, src sent len]
        """
        # Apply self-attention and layer normalization
        src = self.ln(src + self.do(self.sa(src, src, src, src_mask)))
        # Apply positionwise feedforward and layer normalization
        src = self.ln(src + self.do(self.pf(src)))
        
        return src

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, encoder_layer, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.encoder_layer = encoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.device = device

        # Token embedding and position embedding
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)
        
        # Create `n_layers` of encoder layers
        self.layers = nn.ModuleList([encoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device) 
                                     for _ in range(n_layers)])
        
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        """
        params:
        src = [batch size, src sent len]
        src_mask = [batch size, src sent len]

        return: src = [batch size, src sent len, hid dim]
        """
        # initialize position index
        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)
        
        # TODO create position-aware input (token embedding + position embedding). src = [batch size, src sent len, hid dim]
        # scaling token embeddings for numerical stability
        src = self.do((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        # Pass the input through each encoder layer
        for layer in self.layers:
            src = layer(src, src_mask)
        
        # The final output of the encoder src = [batch size, src sent len, hid dim]
        return src

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()
        
        self.ln = nn.LayerNorm(hid_dim)                             # layer norm
        self.sa = self_attention(hid_dim, n_heads, dropout, device) # self-attention
        self.ea = self_attention(hid_dim, n_heads, dropout, device) # encoder-decoder attention
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)  # postiionwise feedforward layer
        self.do = nn.Dropout(dropout)
        
    def forward(self, trg, src, trg_mask, src_mask):
        """
        params:
        trg: target sequence  [batch size, trg sent len, hid dim]
        src: source sequence [batch size, src sent len, hid dim]
        trg_mask: target mask [batch size, trg sent len]
        src_mask: source mask [batch size, src sent len]

        return: trg = [batch size, trg sent len, hid dim]
        """
        # The output of each step is added to the input and then passed 
        
        # first apply self-attention and layer normalization 
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        # next Apply encoder-decoder attention and layer normalization 
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        # finally Apply positionwise feedforward layers and layer normalization 
        trg = self.ln(trg + self.do(self.pf(trg)))
        
        # trg = [batch size, trg sent len, hid dim]
        return trg

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)
        
        self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
                                     for _ in range(n_layers)])
        
        self.fc = nn.Linear(hid_dim, output_dim)
        
        self.do = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, src, trg_mask, src_mask):
        """
        params
        trg = [batch_size, trg sent len]
        src = [batch_size, src sent len]
        trg_mask = [batch size, trg sent len]
        src_mask = [batch size, src sent len]

        return
        logits = [batch size, trg sent len, output dim]
        """
        # initialize position index 
        pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(self.device)
        
        # TODO input is token embedding plus positional embedding
        trg = self.do((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg = [batch size, trg sent len, hid dim]
        
        # Pass the input through each decoder layer
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        
        # Apply the output layer to produce logits
        # logits [batch size, trg sent len, output dim]
        logits = self.fc(trg)
        return logits
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device
        
    def make_masks(self, src, trg):
      """
      creates masks for the source and target sequences.
      The masks are used during self-attention in both the encoder and the decoder
      to prevent the model from attending to padding tokens or future tokens in the target sequence.

      Args:
          src (torch.Tensor): Source sequence tensor of shape [batch size, src sent len]
          trg (torch.Tensor): Target sequence tensor of shape [batch size, trg sent len]

      Returns:
          src_mask (torch.Tensor): Source mask tensor of shape [batch size, 1, 1, src sent len]
          trg_mask (torch.Tensor): Target mask tensor of shape [batch size, 1, trg sent len, trg sent len]
      """
      # Create source mask by checking which tokens are not padding tokens.
      # This mask is used in the encoder self-attention to prevent attending to padding tokens.
      src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

      # Create target padding mask by checking which tokens are not padding tokens.
      trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

      # Create target subsequent mask to prevent attending to future tokens in the target sequence.
      # This is used in the decoder self-attention mechanism.
      # `torch.tril` returns the lower triangular matrix of a matrix 
      trg_len = trg.shape[1]
      trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.uint8, device=self.device))

      # Combine target padding mask and target subsequent mask.
      trg_mask = trg_pad_mask & trg_sub_mask

      return src_mask, trg_mask

    
    def forward(self, src, trg):
        """
        takes source and target sequences as input and returns logits after passing through the encoder and decoder.

        Args:
            src (torch.Tensor): Source sequence tensor of shape [batch size, src sent len].
            trg (torch.Tensor): Target sequence tensor of shape [batch size, trg sent len].

        Returns:
            out (torch.Tensor): Output logits tensor of shape [batch size, trg sent len, output dim].
        """

        # Create masks for the source and target sequences.
        src_mask, trg_mask = self.make_masks(src, trg)

        # compute hidden states of the encoder by Pass the source sequence and source mask through the encoder.
        # These hidden states capture the contextual information of the input source sequence and serve as the input for the decoder
        # enc_src = [batch size, src sent len, hid dim] 
        enc_src = self.encoder(src, src_mask) 
           
        # Pass the target sequence, encoded source sequence, target mask, and source mask through the decoder.
        out = self.decoder(trg, enc_src, trg_mask, src_mask) 
        
        # return logits. out = [batch size, trg sent len, output dim]
        return out


class NoamOpt:
    """
    Noam Optimizer. a learning rate scheduler used in conjunction with the Adam optimizer, 
    specifically designed for training the Transformer model. 
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))