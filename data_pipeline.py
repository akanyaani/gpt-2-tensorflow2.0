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


def _create_min_max_boundaries(
        max_length, min_boundary=_MIN_BOUNDARY, boundary_scale=_BOUNDARY_SCALE):
    bucket_boundaries = []
    x = min_boundary
    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * boundary_scale))

    # Create min and max boundary lists from the initial list.
    buckets_min = [0] + bucket_boundaries
    buckets_max = bucket_boundaries + [max_length + 1]
    return buckets_min, buckets_max


def _get_example_length(example):
    length = tf.shape(example)[0]
    return length


def dynamic_batching(dataset, batch_size, max_length):
    buckets_min, buckets_max = _create_min_max_boundaries(max_length)
    bucket_batch_sizes = [batch_size // x for x in buckets_max]
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    def example_to_bucket_id(example_input):
        seq_length = _get_example_length(example_input)
        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length),
            tf.less(seq_length, buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id):
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        bucket_batch_size = window_size_fn(bucket_id)
        return grouped_dataset.padded_batch(bucket_batch_size, ([None]))

    return dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=None,
        window_size_func=window_size_fn))


def load_vocab(vocab_fpath):
    vocab = collections.OrderedDict()
    index = 0
    for line in open(vocab_fpath, 'r').read().splitlines():
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
                      static_batch=True,
                      max_length=512,
                      padded_shapes=([515]),
                      shuffle=False):
    dataset = make_dataset(tf_records)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    if static_batch:
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    else:
        dataset = dynamic_batching(dataset, batch_size, max_length)
    # dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
