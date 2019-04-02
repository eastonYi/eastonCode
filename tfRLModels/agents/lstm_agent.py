import tensorflow as tf

from tfModels.layers import make_multi_cell
from .agent import Agent


class LSTM_Agent(Agent):
    def __init__(self, is_train, args, name='lstm_agent'):
        self.num_cell_units = args.model.agent.num_cell_units
        self.num_layers = args.model.agent.num_layers
        self.dropout = args.model.agent.dropout
        super().__init__(is_train, args, name)
        self.cell = self.build()

    def build(self):
        cell = make_multi_cell(
            num_cell_units=self.num_cell_units,
            is_train=self.is_train,
            keep_prob=1-self.dropout,
            rnn_mode='BLOCK',
            num_layers=self.num_layers)
        return cell

    def forward(self, state, state_agent):
        """
        from state to action
        """
        with tf.variable_scope(self.name):

            with tf.variable_scope("forward"):
                _output, state_agent = tf.contrib.legacy_seq2seq.rnn_decoder(
                    decoder_inputs=[state],
                    initial_state=state_agent,
                    cell=self.cell)

                cur_logit = tf.layers.dense(
                    inputs=_output[0],
                    units=self.args.dim_output,
                    activation=None,
                    use_bias=False,
                    name='fully_connected')

        return cur_logit, state_agent


    def zero_state(self, batch_size):
        return self.cell.zero_state(batch_size, dtype=tf.float32)
