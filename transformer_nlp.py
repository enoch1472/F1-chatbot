import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math
import requests
from bs4 import BeautifulSoup

# I built this PositionalEncoding layer from scratch to help my transformer 
# understand word order without using recurrence.
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pos_enc = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = np.sin(pos * div_term)
        pos_enc[:, 1::2] = np.cos(pos * div_term)
        pos_enc = pos_enc[np.newaxis, ...]
        self.pos_enc = tf.cast(pos_enc, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_enc[:, :seq_len, :]

# My MultiHeadSelfAttention implementation allows my model to focus 
# on different parts of the F1 sentence structure concurrently.
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        scaled_attention = tf.matmul(attention_weights, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output

class FeedForward(layers.Layer):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.dense1 = layers.Dense(d_ff, activation='relu')
        self.dense2 = layers.Dense(d_model)

    def call(self, x):
        return self.dense2(self.dense1(x))

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training):
        attn_output = self.att(x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# This is my custom 'Tiny Transformer' architecture. I've integrated 
# my custom embeddings and attention layers to create a domain-specific model.
class TinyTransformerLM(keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super(TinyTransformerLM, self).__init__()
        self.token_embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.transformer_layers = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.dropout = layers.Dropout(0.1)
        self.final_layer = layers.Dense(vocab_size)

    def call(self, x, training=False):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        for layer in self.transformer_layers:
            x = layer(x, training=training)
        x = self.dropout(x, training=training)
        logits = self.final_layer(x)
        return logits

def extract_text_from_webpage(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

text_source = "https://en.wikipedia.org/wiki/Formula_One"
text_data = extract_text_from_webpage(text_source)

text_data = text_data.lower()
tokens = text_data.split()[:2500]

vocab = sorted(set(tokens))
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
encoded_text = [word2idx[word] for word in tokens]

sequence_length = 10
inputs = []
targets = []

for i in range(len(encoded_text) - sequence_length):
    inputs.append(encoded_text[i:i+sequence_length])
    targets.append(encoded_text[i+1:i+sequence_length+1])

inputs = np.array(inputs)
targets = np.array(targets)

batch_size = 16
if len(inputs) < batch_size:
    batch_size = max(1, len(inputs))
buffer_size = 10000

dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

vocab_size = len(vocab)
d_model = 128
num_heads = 4
d_ff = 256
num_layers = 2
max_len = sequence_length

model = TinyTransformerLM(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss_object)

epochs = 5
model.fit(dataset, epochs=epochs)

def generate_text(model, start_text, gen_length=20):
    tokens_start = start_text.lower().split()
    input_seq = [word2idx[word] for word in tokens_start if word in word2idx]
    if not input_seq:
        input_seq = [0]
    input_seq = tf.expand_dims(input_seq, 0)
    generated = list(tokens_start)
    last_word = generated[-1] if generated else ""
    for _ in range(gen_length):
        predictions = model(input_seq, training=False)
        last_logits = predictions[0, -1, :]
        predicted_id = tf.random.categorical(tf.expand_dims(last_logits, 0), num_samples=1)[0, 0].numpy()
        new_word = idx2word[predicted_id]
        if new_word != last_word:
            generated.append(new_word)
        last_word = new_word
        input_seq = tf.concat([input_seq, tf.expand_dims([predicted_id], 0)], axis=1)
        if input_seq.shape[1] > sequence_length:
            input_seq = input_seq[:, -sequence_length:]
    return " ".join(generated)

def generate_f1_commentary(start_text):
    return "[Transformer Mode]: " + generate_text(model, start_text, gen_length=20)

# I implemented this sentiment analyzer to detect emotional keywords. 
# This allows the bot to acknowledge the user's tone before generating a response.
def analyze_sentiment(text):
    text = text.lower()
    score = 0
    words = text.split()
    for w in words:
        if w in {'great', 'win', 'champion', 'fast', 'amazing', 'happy', 'love', 'perfect', 'best'}: score += 1
        if w in {'bad', 'crash', 'fail', 'slow', 'sad', 'hate', 'lose', 'worst', 'terrible'}: score -= 1
    if score > 0: return "Positive"
    elif score < 0: return "Negative"
    else: return "Neutral"
