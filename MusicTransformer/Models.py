import tensorflow as tf
import numpy as np


def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

  
# ref : https://github.com/scpark20/music-transformer/blob/master/music-transformer.ipynb
class RelativeGlobalAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(RelativeGlobalAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.headDim = d_model // num_heads
        self.contextDim = int(self.headDim * self.num_heads)

        assert d_model % self.num_heads == 0
        
        self.wq = tf.keras.layers.Dense(self.headDim)
        self.wk = tf.keras.layers.Dense(self.headDim)
        self.wv = tf.keras.layers.Dense(self.headDim)
    
    def call(self, v, k, q, mask):
        # [Heads, Batch, Time, HeadDim]
        q = tf.stack([self.wq(q) for _ in range(self.num_heads)])
        k = tf.stack([self.wk(k) for _ in range(self.num_heads)])
        v = tf.stack([self.wv(v) for _ in range(self.num_heads)])
        print("inputs")
        print("[Heads, Batch, Time, HeadDim]", q.shape)

        self.batch_size = q.shape[1]
        self.max_len = q.shape[2]
        
        #skewing
        # Heads, Time, HeadDim
        E = self.add_weight('E', shape=[self.num_heads, self.max_len, self.headDim]) 
        # [Heads, Batch * Time, HeadDim]
        Q_ = tf.reshape(q, [self.num_heads, self.batch_size * self.max_len, self.headDim])
        # [Heads, Batch * Time, Time]
        S = tf.matmul(Q_, E, transpose_b=True)
        # [Heads, Batch, Time, Time]
        S = tf.reshape(S, [self.num_heads, self.batch_size, self.max_len, self.max_len])
        # [Heads, Batch, Time, Time+1]
        S = tf.pad(S, ((0, 0), (0, 0), (0, 0), (1, 0)))
        # [Heads, Batch, Time+1, Time]
        S = tf.reshape(S, [self.num_heads, self.batch_size, self.max_len + 1, self.max_len])   
        # [Heads, Batch, Time, Time]
        S = S[:, :, 1:]
        # [Heads, Batch, Time, Time]
        attention = (tf.matmul(q, k, transpose_b=True) + S) / np.sqrt(self.depth)
        
        # mask tf 2.0 == tf.linalg.band_part
#         mask = tf.linalg.band_part(tf.ones([self.max_len, self.max_len]), -1, 0)
        attention = attention * mask - tf.cast(1e10, attention.dtype) * (1-mask)
        
        score = tf.nn.softmax(attention, axis=3)
        print("Score : ", score.shape)

        # [Heads, Batch, Time, HeadDim]
        context = tf.matmul(score, v)
        print("[Heads, Batch, Time, HeadDim] : ", context.shape)
        # [Batch, Time, Heads, HeadDim]
        context = tf.transpose(context, [1, 2, 0, 3])
        print("[Batch, Time, Heads, HeadDim] : ", context.shape)        
        # [Batch, Time, ContextDim]
        context = tf.reshape(context, [self.batch_size, self.max_len, self.num_heads * self.headDim])
        print("[Batch, Time, ContextDim] : ", context.shape)
        # [Batch, Time, ContextDim]
        context = tf.keras.layers.Dense(EmbeddingDim, activation='relu')(context)
        print("[Batch, Time, ContextDim] : ", context.shape)     
        # [Batch, Time, EventDim]
        logits = tf.keras.layers.Dense(EventDim)(context)

        return logits
      

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
      
      
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training, 
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2
      
      
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
      
      
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, 
             look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

      
class MusicTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(MusicTransformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, 
             look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
