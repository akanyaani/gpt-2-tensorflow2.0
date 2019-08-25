import click
import glob
from model import *
from data_pipeline import input_fn
import tensorflow as tf
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/model"


@click.command()
@click.option('--num-layers', type=int, default=8, show_default=True, help="decoder layers")
@click.option('--embedding-size', type=int, default=768, show_default=True, help="d_model")
@click.option('--num-heads', type=int, default=8, show_default=True, help="n_head")
@click.option('--dff', type=int, default=3072, show_default=True, help="Filter Size")
@click.option('--max-seq-len', type=int, default=515, show_default=True, help="seq length")
@click.option('--vocab-size', type=int, default=50000, show_default=True, help="embedding vocab size")
@click.option('--optimizer', type=str, default="adam", show_default=True, help="optimizer type")
@click.option('--batch-size', type=int, default=16, show_default=True, help="optimizer type")
@click.option('--learning-rate', type=float, default=0.001, show_default=True, help="learning rate")
@click.option('--distributed', type=bool, default=False, show_default=True, help="distributed training")
def train(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
          optimizer="adam", batch_size=16, learning_rate=1e-3, distributed=False):
    tf_records = glob.glob((_ROOT + "/data/tfrecords/*.tfrecord"))
    if distributed:
        dist_dataset = input_fn(tf_records, batch_size=batch_size)
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        dist_dataset = mirrored_strategy.experimental_distribute_dataset(dist_dataset)
        with mirrored_strategy.scope():

            model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
                         optimizer=optimizer, learning_rate=learning_rate)
            model.creat_optimizer()
            model.create_checkpoint_manager(LOG_DIR)
            model.create_summary_writer(MODEL_DIR)

        model.mirrored_strategy = mirrored_strategy
        model.fit(dist_dataset)
    else:
        dataset = input_fn(tf_records, batch_size=batch_size)
        model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
                     optimizer=optimizer, learning_rate=learning_rate)
        model.creat_optimizer()
        model.create_checkpoint_manager(LOG_DIR)
        model.create_summary_writer(MODEL_DIR)
        model.fit(dataset)
        print("Training Done................")


if __name__ == "__main__":
    train()
