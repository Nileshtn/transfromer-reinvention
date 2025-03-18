import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    """
    A PyTorch module for combining token embeddings and time embeddings with dropout.
    Args:
        embedding_dim (int, optional): The dimensionality of the embedding space. Defaults to 64.
        n_vocal (int, optional): The size of the vocabulary (number of unique tokens). Defaults to 128.
        context_len (int, optional): The length of the context (sequence length). Defaults to 50.
    Attributes:
        token_embedding (nn.Embedding): Embedding layer for token embeddings with shape (vocab_size, embedding_dim).
        time_embedding (nn.Parameter): Learnable parameter for time embeddings with shape (embedding_dim, context_len).
        dropout (nn.Dropout): Dropout layer applied to token embeddings with a dropout probability of 0.5.
    Methods:
        forward(x):
            Computes the combined embeddings for the input tokens and time steps.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, context_len), where each element is a token index.
            Returns:
                torch.Tensor: Combined embeddings of shape (batch_size, context_len, embedding_dim).
    """
    def __init__(self, embedding_dim=64, n_vocal=128, context_len=50):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocal, embedding_dim) # (vocab_size, embedding_dim)
        self.time_embedding = nn.Parameter(torch.randn((embedding_dim, context_len))) # (embedding_dim, context_len)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch, context_len = x.shape
        time_range = torch.arange(context_len, device=x.device)
        time_embed = self.time_embedding[:, time_range].unsqueeze(0).expand(batch, -1, -1)
        token_embed = self.token_embedding(x)
        teken_embed = self.dropout(token_embed)
        return time_embed.permute(0, 2, 1) + token_embed  # (batch, context_len, embedding_dim)

class Head(nn.Module):
    """
    A PyTorch module implementing a single attention head for use in transformer models.
    Args:
        context_len (int, optional): The maximum length of the input sequence. Defaults to 50.
        embedding_dim (int, optional): The dimensionality of the input embeddings. Defaults to 64.
        attention_dim (int, optional): The dimensionality of the attention mechanism. Defaults to 8.
        mode (str, optional): The mode of the attention head, either "encoder" or "decoder". 
                              If "encoder", no masking is applied. If "decoder", causal masking is applied. Defaults to "encoder".
    Attributes:
        query (nn.Linear): Linear layer to compute the query vectors.
        value (nn.Linear): Linear layer to compute the value vectors.
        key (nn.Linear): Linear layer to compute the key vectors.
        mode (str): The mode of the attention head ("encoder" or "decoder").
        dropout (nn.Dropout): Dropout layer applied to the attention weights.
        tril (torch.Tensor): A lower triangular matrix used for causal masking in decoder mode.
    Methods:
        forward(x):
            Computes the attention output for the input tensor.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, sequence_length, attention_dim).
    """
    def __init__(self, context_len=50, embedding_dim=64, attention_dim=8, mode="encoder"):
        super().__init__()
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.mode = mode
        self.dropout = nn.Dropout(0.5)
        self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len)))

    def forward(self, x):
        b, t, c = x.shape  
        q = self.query(x)  
        k = self.key(x)    
        v = self.value(x)  
        d_k = q.shape[-1]
        scaled_scores = torch.bmm(q, k.permute(0, 2, 1)) / (d_k ** 0.5)

        if self.mode == "encoder":
            attn_weights = scaled_scores
        else:
            attn_weights = scaled_scores.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)  
        attn_weights = self.dropout(attn_weights)
        attention_output = torch.bmm(attn_weights, v)  
        return attention_output  

class SelfAttention(nn.Module):
    """
    Implements a multi-head self-attention mechanism.
    Attributes:
        attention_dim (int): The dimensionality of each attention head, calculated as embedding_dim // heads.
        heads (nn.ModuleList): A list of attention heads, each implemented as an instance of the `Head` class.
        projection_weight (nn.Linear): A linear layer to project the concatenated outputs of all attention heads back to the original embedding dimension.
        dropout (nn.Dropout): A dropout layer applied after the projection to prevent overfitting.
    Args:
        context_l (int): The length of the input context (sequence length). Default is 50.
        embedding_dim (int): The dimensionality of the input embeddings. Default is 64.
        heads (int): The number of attention heads. Default is 8.
        mode (str): The mode of operation, typically "decoder" or "encoder". Default is "decoder".
    Methods:
        forward(x):
            Computes the forward pass of the self-attention mechanism.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
            Returns:
                torch.Tensor: Output tensor of the same shape as the input, with self-attention applied.
    """
    def __init__(self, context_l=50, embedding_dim=64, heads=8, mode="decoder"):
        super().__init__()
        self.attention_dim = embedding_dim // heads
        self.heads = nn.ModuleList([
            Head(context_len=context_l, embedding_dim=embedding_dim, attention_dim=self.attention_dim, mode=mode)
            for _ in range(heads)
        ])
        self.projection_weight = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        b, t, c = x.shape
        x_ln = F.layer_norm(x, (c,))  # Correct normalization
        output = [head(x_ln) for head in self.heads]
        output = torch.cat(output, dim=2)
        x_p = self.projection_weight(output)
        x_p = self.dropout(x_p)
        return x + x_p

class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) module with residual connections, layer normalization, 
    GELU activation, and dropout. This module is designed to process input tensors 
    with a specified embedding dimension and expand them using a multiplier.
    Attributes:
        mlp_weight (nn.Linear): A linear layer that expands the input embedding dimension 
            by the specified multiplier.
        mlp_projection (nn.Linear): A linear layer that projects the expanded dimension 
            back to the original embedding dimension.
        gelu1 (nn.GELU): GELU activation function applied after the first linear layer.
        dropout (nn.Dropout): Dropout layer applied after the activation function.
    Args:
        embedding_dim (int, optional): The size of the input embedding dimension. 
            Defaults to 64.
        mlp_multiplier (int, optional): The multiplier used to expand the embedding 
            dimension in the first linear layer. Defaults to 4.
    Methods:
        forward(x):
            Applies the MLP transformation to the input tensor `x` with residual 
            connections. The input tensor is normalized, passed through the MLP layers, 
            and added back to the original input.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
            Returns:
                torch.Tensor: Output tensor of the same shape as the input.
    """
    def __init__(self, embedding_dim=64, mlp_multiplier=4):
        super().__init__()
        self.mlp_weight = nn.Linear(embedding_dim, embedding_dim * mlp_multiplier)
        self.mlp_projection = nn.Linear(embedding_dim * mlp_multiplier, embedding_dim)
        self.gelu1 = nn.GELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        b, t, c = x.shape
        x_ln = F.layer_norm(x, (c,))
        output = self.gelu1(self.mlp_weight(x_ln))
        output = self.dropout(output)
        output = self.mlp_projection(output)
        return x + output

class Logits(nn.Module):
    """
    A PyTorch module that computes logits from input embeddings using a linear transformation.
    Args:
        n_vocal (int, optional): The size of the vocabulary (number of output classes). 
                                 Defaults to 128.
        embedding_dim (int, optional): The dimensionality of the input embeddings. 
                                       Defaults to 64.
    Methods:
        forward(x):
            Applies layer normalization to the input tensor and computes logits 
            using a linear transformation.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, sequence_length, n_vocal), 
                              containing the logits for each class.
    """
    def __init__(self, n_vocal=128, embedding_dim=64):
        super().__init__()
        self.encode = nn.Linear(embedding_dim, n_vocal, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        x_ln = F.layer_norm(x, (c,))  # Normalize across channels (embedding dim)
        logits = self.encode(x_ln)  # (b, t, n_vocal)
        return logits  # No softmax applied

class TransformerMini(nn.Module):
    """
    A mini Transformer model implementation designed for sequence processing tasks.
    Args:
        context_l (int): The length of the input context or sequence. Default is 50.
        n_vocal (int): The size of the vocabulary or number of unique tokens. Default is 128.
        embedding_dim (int): The dimensionality of the embedding space. Default is 64.
        attention_heads (int): The number of attention heads in the self-attention mechanism. Default is 8.
        mode (str): The mode of operation for the self-attention mechanism. Default is "decoder".
    Attributes:
        embedding (Embedding): Embedding layer that converts input tokens into dense vectors.
        self_attention (SelfAttention): Self-attention mechanism for capturing dependencies in the sequence.
        mlp (MLP): Multi-layer perceptron for further processing of the sequence representations.
        logits (Logits): Final layer that maps the processed representations to output logits.
    Methods:
        forward(x):
            Processes the input sequence through the embedding, self-attention, MLP, and logits layers.
            Args:
                x (Tensor): Input tensor of shape (batch, context_len).
            Returns:
                Tensor: Output tensor of shape (batch, context_len, n_vocal), representing the logits for each token.
    """
    def __init__(self, context_l=50, n_vocal=128, embedding_dim=64, attention_heads=8, mode="decoder"):
        super().__init__()
        self.embedding = Embedding(embedding_dim, n_vocal, context_l)
        self.self_attention = SelfAttention(context_l, embedding_dim, attention_heads, mode)
        self.mlp = MLP(embedding_dim)
        self.logits = Logits(n_vocal, embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.self_attention(x)
        x = self.mlp(x)
        x = self.logits(x)
        return x  # Output shape: (batch, context_len, n_vocal)