from sample import SequenceGenerator
import click


@click.command()
@click.option('--seq-len', type=int, default=512, show_default=True, help="seq_len")
@click.option('--temperature', type=float, default=1.0, show_default=True, help="seq_len")
@click.option('--top-k', type=int, default=512, show_default=True, help="seq_len")
@click.option('--top-p', type=int, default=512, show_default=True, help="seq_len")
@click.option('--nucleus_sampling', type=int, default=512, show_default=True, help="seq_len")
def seq_gen(seq_len, temperature, top_k, top_p, nucleus_sampling):
    sg = SequenceGenerator(model_path, model_param, bpe_data_path)
    sg.load_weights()
    sg.sample_sequence(context,
                       seq_len=512,
                       temperature=1,
                       top_k=8,
                       top_p=0.9,
                       nucleus_sampling=True)


if __name__ == "__main__":
    seq_gen()
