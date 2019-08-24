import tensorflow as tf
import math


class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, vocab_size, hidden_size, stddev=0.01, mean=0.0):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.stddev = stddev
        self.mean = mean

    def build(self, input_shape):
        with tf.name_scope("embedding_layer"):
            self.shared_weights = self.add_weight(
                "weights",
                shape=[self.vocab_size, self.hidden_size],
                dtype="float32",
                initializer=tf.random_normal_initializer(mean=self.mean, stddev=self.stddev)
            )
        # tf.summary.histogram("embedding_wights", self.shared_weights, step=tf.summary.experimental.get_step())
        super(EmbeddingLayer, self).build(input_shape)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
        }

    def call(self, inputs, scale=False):
        with tf.name_scope("embedding"):
            # Create binary mask of size [batch_size, length]
            mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
            inputs = tf.cast(inputs, tf.int32)
            # embeddings = tf.gather(self.shared_weights, inputs)
            embeddings = tf.nn.embedding_lookup(self.shared_weights, inputs)
            embeddings *= tf.expand_dims(mask, -1)
            # Scale embedding by the sqrt of the hidden size
            if scale:
                embeddings *= self.hidden_size ** 0.5

            return embeddings


class PositionEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, max_seq_len, hidden_size, trainable=True, stddev=0.02, mean=0.0):
        super(PositionEmbeddingLayer, self).__init__()
        self.position_seq = max_seq_len
        self.hidden_size = hidden_size
        self.trainable = trainable
        self.stddev = stddev
        self.mean = mean

        if trainable:
            self.position_embedding = EmbeddingLayer(self.position_seq, self.hidden_size,
                                                     stddev=self.stddev, mean=self.mean)

    def get_config(self):
        return {
            "seq_len": self.position_seq,
            "hidden_size": self.hidden_size,
            "trainable": self.trainable
        }

    def call(self, inputs, start=1):
        with tf.name_scope("pos_embedding"):
            if self.trainable:
                self.batch_size = tf.shape(inputs)[0]
                self.batch_seq = tf.shape(inputs)[1]

                positions = tf.reshape(tf.tile(tf.range(start, self.batch_seq + start), [self.batch_size]),
                                       [self.batch_size, self.batch_seq])

                positions = tf.cast(positions, tf.int32)
                position_mask = tf.cast(tf.not_equal(inputs, 0), tf.int32)
                positions *= position_mask

                return self.position_embedding(positions)
            else:
                return self.get_position_sinusoid(self.seq_len)

    @staticmethod
    def get_position_sinusoid(seq_len, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
        position = tf.cast(tf.range(seq_len), tf.float32)
        num_timescales = hidden_size // 2
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.cast(num_timescales, tf.float32) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        return signal
