# We choose to take the encoder from the Attention Is All You Need paper and add a dense head for prediction;
# TensorFlow has a tutorial on creating the full encoder-decoder at
#     https://www.tensorflow.org/text/tutorials/transformer
# so we can just grab the functions which we need from there and modify them to suit our purposes

import tensorflow as tf
import numpy as np

def positional_encoding(length, depth) :
    '''
    Creates positional encodings for sequence of length
    length using sin curves sampled at depth points.
    '''
    
    depth = depth/2
    
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )
    
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer) :
    '''
    Custom keras layer to perform embedding and
    adding positional encoding to sequence inputs.
    '''
    
    def __init__(self, vocab_size=34, d_model=256, max_length=99) :
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=max_length, depth=d_model)
        
    def compute_mask(self, *args, **kwargs) :
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, x) :
        length = tf.shape(x)[1]
        x = self.embedding(x)
        
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        p = self.pos_encoding[tf.newaxis, :length, :]
        x = x + p
        return x
    
class SelfAttention(tf.keras.layers.Layer) :
    '''
    Performs multi-head self-attention followed
    by a residual connection and a layer normalization.
    '''
    
    def __init__(self, num_heads=2, key_dim=256) :
        
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
    def call(self, x) :
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layer_norm(x)
        return x
    
class FeedForward(tf.keras.layers.Layer) :
    '''
    A feed forward network with on hidden layer, ReLU
    activation, and dropout. Ends with layer norm.
    '''
    
    def __init__(self, d_model, dff, dropout_rate=0.1) :
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
    def call(self, x) :
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x
    
class EncoderLayer(tf.keras.layers.Layer) :
    '''
    One layer in the encoder. Perfoms multi-head
    self-attention followed by a feed forward network.
    '''
    
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1) :
        
        super().__init__()
        
        self.self_attention = SelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
        )
        
        self.ffn = FeedForward(d_model, dff, dropout_rate=dropout_rate)
        
    def call(self, x) :
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    
class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, num_layers, d_model, num_heads,
                 dff, vocab_size=34, dropout_rate=0.1, max_length=99):
        
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length
        )
        
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, x) :
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        
        # Add dropout.
        x = self.dropout(x)
        
        for i in range(self.num_layers) :
            x = self.enc_layers[i](x)
            
        return x  # Shape `(batch_size, seq_len, d_model)`.
    
def Transformer(
    num_layers,
    d_model,
    num_heads,
    dff,
    dense_hdim=124,
    num_classes=4,
    max_length=99,
    vocab_size=34,
    dropout_rate=0.1
) :
    
    # inputs will have fixed length
    inp = tf.keras.Input((max_length))
    
    # apply encoder
    enc = Encoder(num_layers,
                  d_model,
                  num_heads,
                  dff,
                  vocab_size=vocab_size,
                  dropout_rate=dropout_rate,
                  max_length=max_length
                 )
    enc_output = enc(inp)
    
    # classifier will be fully-connected with one hidden layer
    # applied only to first token (the special class token)
    #enc_cls = enc_output[:, 0]
    enc_cls = tf.keras.layers.GlobalAveragePooling1D()(enc_output)
    H = tf.keras.layers.Dense(dense_hdim, activation='relu')(enc_cls)
    P = tf.keras.layers.Dense(num_classes)(H)
    if num_classes == 1 :
        P = tf.squeeze(P)
    
    return tf.keras.Model(inp, P)