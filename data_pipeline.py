import collections

import tensorflow as tf

PAD_ID = 0
UNKNOWN_ID = 1
START_ID = 2
END_ID = 3
_READ_RECORD_BUFFER = 8 * 1000 * 1000
NO_THREADS = 16

_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


def load_vocab(vocab_path):
    vocab = collections.OrderedDict()
    index = 0
    for line in open(vocab_path, 'r').read().splitlines():
        vocab[line.split()[0]] = index
        index += 1
    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab


def convert_by_vocab(vocab, items):
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def load_tf_records(filenames):
    if type(filenames) is str:
        filenames = [filenames]
    # print(filenames)
    return tf.data.TFRecordDataset(filenames, buffer_size=_READ_RECORD_BUFFER)


def parse_example(serialized_example):
    data_fields = {
        "inputs": tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)
    inputs = tf.sparse.to_dense(parsed["inputs"])

    inputs = tf.cast(inputs, tf.int32)

    return inputs


def make_dataset(tf_files, no_threads=NO_THREADS):
    dataset = load_tf_records(tf_files)
    dataset = dataset.map(parse_example, num_parallel_calls=no_threads)
    return dataset


def tf_batch_iterator(tf_records,
                      batch_size=64,
                      num_epochs=10,
                      max_length=512,
                      padded_shapes=([-1]),
                      shuffle=False):
    dataset = make_dataset(tf_records)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
