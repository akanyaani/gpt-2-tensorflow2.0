import glob
import json

import click

from data_pipeline import input_fn
from gpt2_model import *

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/model"


@click.command()
@click.option('--num-layers', type=int, default=8, show_default=True, help="No. of decoder layers")
@click.option('--embedding-size', type=int, default=768, show_default=True, help="Embedding size")
@click.option('--num-heads', type=int, default=8, show_default=True, help="Number of heads")
@click.option('--dff', type=int, default=3072, show_default=True, help="Filter Size")
@click.option('--max-seq-len', type=int, default=515, show_default=True, help="Seq length")
@click.option('--vocab-size', type=int, default=24512, show_default=True, help="Vocab size")
@click.option('--optimizer', type=str, default="adam", show_default=True, help="optimizer type")
@click.option('--batch-size', type=int, default=8, show_default=True, help="optimizer type")
@click.option('--learning-rate', type=float, default=0.001, show_default=True, help="learning rate")
@click.option('--graph-mode', type=bool, default=False, show_default=False, help="TF run mode")
@click.option('--distributed', type=bool, default=False, show_default=False, help="distributed training")
def train(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
          optimizer, batch_size, learning_rate, graph_mode, distributed):
	par_map = {"num_layers": num_layers, "d_model": embedding_size,
	           "num_heads": num_heads, "dff": dff,
	           "max_seq_len": max_seq_len, "vocab_size": vocab_size}

	# exp_name = "_".join(['{}_{}'.format(k, v) for k, v in par_map.items()])

	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)

	with open(MODEL_DIR + '/model_par.json', 'w') as f:
		json.dump(par_map, f)

	tf_records = glob.glob((_ROOT + "/data/tf_records/*.tfrecord"))
	train_percent = int(len(tf_records) * (85 / 100))

	print("No. of tf records:- ", len(tf_records))
	train_tf_records = tf_records[:train_percent]
	test_tf_records = tf_records[train_percent:]

	train_dataset = input_fn(train_tf_records, batch_size=batch_size)
	test_dataset = input_fn(test_tf_records, batch_size=batch_size)

	if distributed:
		mirrored_strategy = tf.distribute.MirroredStrategy()
		train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
		test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset)

		with mirrored_strategy.scope():

			model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
			             optimizer=optimizer, learning_rate=learning_rate)
			model.create_optimizer()
			model.create_checkpoint_manager(MODEL_DIR)
			model.create_summary_writer(LOG_DIR)

		model.mirrored_strategy = mirrored_strategy
		model.global_batch_size = tf.cast(batch_size, tf.float32)
	else:
		model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
		             optimizer=optimizer, learning_rate=learning_rate)
		model.create_optimizer()
		model.create_checkpoint_manager(MODEL_DIR)
		model.create_summary_writer(LOG_DIR)

	model.fit([train_dataset, test_dataset], graph_mode)
	print("Training Done................")


if __name__ == "__main__":
	train()
