import tensorflow as tf

from .agent import Agent


class Q_Estimator(Agent):
    def __init__(self, size, is_train, args, name='q_estimator'):
        self.q_estimation = tf.ones(size) * 21
        self.UCB_param = args.agent.UCB_param
        self.action_count = tf.zeros([size])
        self.choose = tf.random.uniform([], minval=0, maxval=None)
        super().__init__(is_train, args, name)

    def forward(self):
        """
        from state to action
        """
        # action
        if self.UCB_param:
            UCB_estimation = self.q_estimation + \
                self.UCB_param * tf.sqrt(tf.log(self.time + 1) / (self.agent.action_count + 1e-5))
            q_best = tf.reduce_max(UCB_estimation)
            # return np.random.choice([action for action, q in enumerate(UCB_estimation) if q == q_best]
        else:
            action = tf.cond(
                tf.less(self.choose, self.args.agent.epsilon),
                lambda: self.exploitation(self.q_estimation),
                lambda: self.exploration(self.q_estimation)
            )

        return action

    def exploitation(self, q_estimation):
        if self.args.agent.gradient:
            return tf.distributions.Categorical(logits=q_estimation).sample()
        else:
            return tf.argmax(self.q_estimation, output_type=tf.int32)
