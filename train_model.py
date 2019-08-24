import click
import glob
from model import *
from data_pipeline import tf_batch_iterator
import tensorflow as tf


@click.command()
@click.option('--num-layers', type=int, default=8, show_default=True, help="decoder layers")
@click.option('--d-model', type=int, default=768, show_default=True, help="d_model")
@click.option('--num-heads', type=int, default=8, show_default=True, help="n_head")
@click.option('--dff', type=int, default=3072, show_default=True, help="Filter Size")
@click.option('--max-seq-len', type=int, default=515, show_default=True, help="seq length")
@click.option('--vocab-size', type=int, default=50000, show_default=True, help="embedding vocab size")
@click.option('--data-path', type=str, default="/data/tf_transformer_jd_data", show_default=True,
              help="tf records path")
@click.option('--optimizer', type=str, default="adam", show_default=True, help="optimizer type")
@click.option('--batch-size', type=int, default=16, show_default=True, help="optimizer type")
@click.option('--static-batch', type=bool, default=False, show_default=True, help="static batching")
@click.option('--learning-rate', type=float, default=0.001, show_default=True, help="learning rate")
@click.option('--distributed', type=bool, default=False, show_default=True, help="distributed training")
def train(num_layers, d_model, num_heads, dff, max_seq_len, vocab_size, data_path,
          optimizer="adam", batch_size=16, static_batch=False, learning_rate=1e-3, distributed=False):
    tf_records = data_path + "/*.tfrecord"

    tf_records = glob.glob(tf_records)
    # print(tf_records)
    if distributed:
        print("---------------" + str(batch_size))
        dist_dataset = tf_batch_iterator(tf_records, batch_size=batch_size)
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        dist_dataset = mirrored_strategy.experimental_distribute_dataset(dist_dataset)
        with mirrored_strategy.scope():

            model = Gpt(num_layers, d_model, num_heads, dff, max_seq_len, vocab_size,
                        optimizer=optimizer, learning_rate=learning_rate)
            model.creat_optimizer()
            model.create_checkpoint_manager("../log")
            model.create_summary_writer("../log")

        model.mirrored_strategy = mirrored_strategy
        model.fit(dist_dataset)
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                # tf.config.experimental.set_memory_growth(gpus[0], True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
        dataset = tf_batch_iterator(tf_records, batch_size=batch_size)
        # dataset = np.random.randint(500, size=(200, 20))
        model = Gpt(num_layers, d_model, num_heads, dff, max_seq_len, vocab_size,
                    optimizer=optimizer, learning_rate=learning_rate)

        model.creat_optimizer()
        model.create_checkpoint_manager("../log")
        model.create_summary_writer("../log")
        model.fit(dataset)
        print("Training Done................")


if __name__ == "__main__":
    train()
