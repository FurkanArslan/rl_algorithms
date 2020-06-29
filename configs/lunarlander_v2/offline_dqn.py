"""Config for DQN on LunarLander-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
from rl_algorithms.common.helper_functions import identity

agent = dict(
    type="OfflineDQNAgent",
    hyper_params=dict(
        gamma=0.99,
        tau=5e-3,
        batch_size=256,  # openai baselines: 32
        multiple_update=1,  # multiple learning updates
        from_disk=False,
        experience_dir="data/experience/LunarLander-v2/200626_103729/",
        gradient_clip=10.0,  # dueling: 10.0
        n_step=3,
        w_n_step=1.0,
        w_q_reg=1e-7,
        loss_type=dict(type="IQNLoss"),
    ),
    learner_cfg=dict(
        type="OfflineDQNLearner",
        backbone=dict(),
        head=dict(
            type="IQNMLP",
            configs=dict(
                hidden_sizes=[128, 64],
                use_noisy_net=False,
                n_tau_samples=16,
                n_tau_prime_samples=16,
                n_quantile_samples=32,
                quantile_embedding_dim=64,
                kappa=1.0,
                output_activation=identity,
            ),
        ),
        optim_cfg=dict(lr_dqn=1e-4, weight_decay=1e-7, adam_eps=1e-8),
    ),
)
