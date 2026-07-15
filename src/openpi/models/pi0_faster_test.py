import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config
from openpi.models import pi0_faster


def test_block_causal_action_mask_matrix():
    mask = pi0_faster.make_block_causal_action_mask(action_horizon=9, block_size=3)
    expected = jnp.array(
        [
            [True, True, True, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False, False],
            [True, True, True, True, True, True, False, False, False],
            [True, True, True, True, True, True, False, False, False],
            [True, True, True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
        ],
        dtype=jnp.bool_,
    )

    print(np.where(np.asarray(mask), "0", "-inf"))
    np.testing.assert_array_equal(np.asarray(mask), np.asarray(expected))


def test_sample_block_timesteps_are_blockwise():
    action_time, block_time = pi0_faster.sample_block_timesteps(
        jax.random.key(0), batch_size=4, action_horizon=10, block_size=3
    )

    assert action_time.shape == (4, 10)
    assert block_time.shape == (4, 4)
    np.testing.assert_allclose(np.asarray(action_time[:, 0:3]), np.repeat(np.asarray(block_time[:, 0:1]), 3, axis=1))
    np.testing.assert_allclose(np.asarray(action_time[:, 3:6]), np.repeat(np.asarray(block_time[:, 1:2]), 3, axis=1))
    np.testing.assert_allclose(np.asarray(action_time[:, 6:9]), np.repeat(np.asarray(block_time[:, 2:3]), 3, axis=1))
    np.testing.assert_allclose(np.asarray(action_time[:, 9:10]), np.asarray(block_time[:, 3:4]))
    assert not np.allclose(np.asarray(block_time[:, 0]), np.asarray(block_time[:, 1]))


def test_pi0_faster_block_causal_forcing_fake_forward():
    key = jax.random.key(0)
    config = pi0_config.Pi0FasterConfig(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=9,
        training_mode="block_causal_forcing",
        block_size=3,
    )
    model = config.create(key)

    obs, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)
    loss = model.compute_loss(key, obs, actions, train=True)

    assert loss.shape == (2, 9)
    assert jnp.all(jnp.isfinite(loss))
