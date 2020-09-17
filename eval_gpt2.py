import click
from gpt2_model import *
import tensorflow as tf
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/model"


@click.command()
@click.option('--distributed', type=bool, default=False, show_default=True, help="distributed training")
def train():

    model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
                 optimizer=optimizer, learning_rate=learning_rate)
    model.create_optimizer()
    model.create_checkpoint_manager(MODEL_DIR)
    model.create_summary_writer(LOG_DIR)
    print("Training Done................")


if __name__ == "__main__":
    train()
