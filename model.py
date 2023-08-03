import tensorflow as tf
from keras.layers import Input, Dense, Dropout, LSTM, Embedding
from keras.models import Model
from keras.utils import plot_model

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, dff, rate):
        super(AttentionLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, dff)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(dff, activation='relu'),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

def define_model(total_words, max_length, num_heads=8, dff=256, rate=0.1):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    fe2_expanded = tf.expand_dims(fe2, axis=1)
    fe2_expanded = tf.tile(fe2_expanded, [1, max_length, 1])

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(total_words, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)

    add_layer = tf.keras.layers.add([fe2_expanded, se2])

    attention_layer = AttentionLayer(num_heads, dff, rate)
    attention_output = attention_layer(add_layer)

    se3 = LSTM(256)(attention_output)

    decoder1 = Dense(256, activation='relu')(se3)
    outputs = Dense(total_words, activation='softmax')(decoder1)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)

    return model