import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_model : int, vocab_size: int):
        r"""
        Description:
            Input embedding for x
        vocab_size (int):
            size of the dictionary of embeddings
        d_model (int):
            the size of each embedding vector
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        r"""
        Shape:
        - Input: :(*), IntTensor or LongTensor of arbitrary shape containing the indices to extract
        - Output: (*, H), where * is the input shape and H = embedding_dim.
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int, dropout: float):
        """
        Description:
            Add positional encodding to represent the position of words in a sentence.
        Args:
            d_model:      dimension of embeddings
            dropout:      randomly zeroes-out some of the input
            max_length:   max sequence length
        """

        super().__init__()
        self.d_model = d_model
        self.seq_len = max_length
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(max_length, d_model)

        # create a vector of shape (seq_length, 1)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(1e4)/d_model))
        # apply the sin to even position
        pe[:,0::2]  = torch.sin(position*div_term)
        # apply the cos to even position
        pe[:,1::2]  = torch.cos(position*div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model), add batch dimension

        self.register_buffer('pe',pe)
    
    def forward(self, x):
        """
        Args:
            x: embeddings (batch_size, seq_length, d_model)
        Returns:
            embeddings + positional encodings (batch_size, seq_length, d_model)
        """

        # add positional encoding to the embedding. x.shape[1] is seq_length, and if seq_length<max_seq_length, `:x.shape[1]` ensure to pick first seq_length.
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Unlernlable param
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, features: int,  eps: float = 1e-6):
        """
        Args:
            eps: small variance added to avoid division by zeros
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) #multiplied, learnable param
        self.bias = nn.Parameter(torch.zeros(features)) #added, learnable param

    def forward(self,x):
        """
        Args:
            x: (batch_size, seq_length, d_model)
        Returns:
            x (batch_size, seq_length, d_model), normalize across feature dimension, scaled by two learnable param. 
        """
        mean = x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
        std  = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        return self.alpha + (x - mean)/(std + self.eps) + self.bias # eps is to prevent dividing by zero or when std is very small
    


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Args:
            d_model: The feature dimension
            d_ff : size of intermidate layer
            dropout: dropout probability
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, d_model)
        InBetween:
            # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_ff) --> (batch_size, seq_length, d_model)
        Returns:
            x (batch_size, seq_length, d_model)
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        """
        Description:
            Calculate self/cross multi-head attention. 
        Args:
            d_model : The feature dimension/dimension of embeddings
            h : number of attention heads
            dropout: dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv

        self.w_o = nn.Linear(d_model, d_model) #Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout : nn.Dropout):
        """
        Description:
            Static method to calculate attention score.
        Args:
            query : shape (batch_size, seq_length, d_model)
            key :   shape (batch_size, seq_length, d_model)
            value:  shape (batch_size, seq_length, d_model)
            mask:   shape (batch_size, seq_length, d_model), for self attension used to avoid calulate attention score with padding token
            dropout :  dropout probability
        Return:
            (attention_scores @ value , attention_scores)
        """
        d_k = query.shape[-1]

        # (batch_size, h, seq_length, seq_length) --> # (batch_size, h, seq_length, seq_length)
        attention_scores = (query@key.transpose(-2,-1))/math.sqrt(d_k)  #transpose key from (batch_size, h, seq_length, d_k) --> (batch_size, h, d_k, seq_length)
        
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (batch_size, h, seq_length, seq_length)
    
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value , attention_scores

    def forward(self, q, k, v, mask):
        """
        Description:
            Method to output attention score.
        Args:
            query : shape (batch_size, seq_length, d_model)
            key :   shape (batch_size, seq_length, d_model)
            value:  shape (batch_size, seq_length, d_model)
            mask:   shape (batch_size, seq_length, d_model), for self attension used to avoid calulate attention score with padding token
        Return:
            
        """
        query = self.w_q(q) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        key = self.w_k(k)   # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        value = self.w_v(v) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)

        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, h, d_k) -- transpose --> (batch_size, h, seq_length, d_k)
        # Why transpose? this way all the heads will see all the sentence, but will see diff embedding for same sentence.
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key  = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_length, d_k) --> (batch_size, seq_length, h, d_k) --> (batch_size, seq_length, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch_size, h, seq_length, d_k) --> (batch_size, h, seq_length, d_k) 
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def  __init__(self, features: int, dropout : float):
        """
        Description:
            residul connection. x = x + h(x)
        Args:
            features : num_features
            dropout :  dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm  = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        """
        Args:
            X : input
            sublayer : NN block (h) from which x passes through
        return:
            x + h(x)
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout: float):
        """
        Description:
           One encoder block
        Args:
            features : num_features
            self_attention_block : MultiHeadAttentionBlock
            feed_forward_block : feed_forward_block
            dropout : dropout probability
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        """
        Args:
            x : input
            src_mask : mask used to mask some value in calulating attention scores
        return:
            output of a single encoder block
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # calculate self_sttention. Key, value, query are all equal to input x.
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        """
        Description:
            Encoder module of transformer
        Args:
            features : num_features
            layers : ModuleList where each item is an encoder block
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, mask):
        """
        Args:
            x : input
            mask :  mask used to mask some value in calulating attention scores
        retuens:
            output of transformer encoder module
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block : MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout: float):
        """
        Initialize a DecoderBlock.

        Args:
            features : num_features
            self_attention_block (MultiHeadAttentionBlock): A block for self-attention.
            cross_attention_block (MultiHeadAttentionBlock): A block for cross-attention with encoder output.
            feed_forward_block (FeedForwardBlock): A block for feed-forward processing.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass of the DecoderBlock.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask for cross-attention.
            tgt_mask (torch.Tensor): Target mask for self-attention.

        Returns:
            torch.Tensor: Output tensor after processing through the block.
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # calculate self_sttention. Key, value, query are all equal to x, but since its a decoder, we will use tgt_mask
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        """
        Initialize a Decoder.

        Args:
            features : num_features
            layers (nn.ModuleList): List of DecoderBlocks.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass of the Decoder.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask for cross-attention.
            tgt_mask (torch.Tensor): Target mask for self-attention.

        Returns:
            torch.Tensor: Output tensor after processing through the decoder.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize a ProjectionLayer.

        Args:
            d_model (int): Dimension of the model.
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        Forward pass of the ProjectionLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Log-softmax projection of the input.
        """
        # (batch_size, seq_length, d_model) --> # (batch_size, seq_length, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder : Decoder, src_embed : InputEmbedding, tgt_embed : InputEmbedding, src_pos: PositionalEncoding, tgt_pos : PositionalEncoding, projection_layer = ProjectionLayer):
        """
        Initialize a Transformer model.

        Args:
            encoder (Encoder): The encoder module.
            decoder (Decoder): The decoder module.
            src_embed (InputEmbedding): Source embedding module.
            tgt_embed (InputEmbedding): Target embedding module.
            src_pos (PositionalEncoding): Source positional encoding module.
            tgt_pos (PositionalEncoding): Target positional encoding module.
            projection_layer (ProjectionLayer): Projection layer for output (optional).
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        """
        Encode the source sequence.

        Args:
            src (torch.Tensor): Source sequence.
            src_mask (torch.Tensor): Source mask.

        Returns:
            torch.Tensor: Encoded source sequence.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        Decode module.

        Args:
            encoder_output (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask for cross-attention.
            tgt (torch.Tensor): Target sequence.
            tgt_mask (torch.Tensor): Target mask for self-attention.

        Returns:
            torch.Tensor: Decoded target sequence.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """
        Project the output tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Projected output.
        """
        return self.projection_layer(x)
    


def build_transformer(src_vocab_size : int, tgt_vocab_size : int, src_seq_len : int, tgt_seq_len : int, d_model : int = 512, N : int = 6, h: int = 8, dropout : float = 0.1, d_ff : int = 2048) -> Transformer:

    """
    Build a Transformer model with customizable configuration.

    Args:
        src_vocab_size (int): The vocabulary size of the source language.
        tgt_vocab_size (int): The vocabulary size of the target language.
        src_seq_len (int): Maximum sequence length for the source language.
        tgt_seq_len (int): Maximum sequence length for the target language.
        d_model (int, optional): The dimensionality of the model (default is 512).
        N (int, optional): The number of encoder and decoder blocks (default is 6).
        h (int, optional): The number of attention heads (default is 8).
        dropout (float, optional): The dropout probability (default is 0.1).
        d_ff (int, optional): The dimension of the feed-forward layer (default is 2048).

    Returns:
        Transformer: A configured Transformer model based on the provided parameters.
    """

    # create the embedding layer
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # create pos layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout) # don't need seperate encoding, can define one with max(src_seq_len, tgt_seq_len), as they are constant


    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer



