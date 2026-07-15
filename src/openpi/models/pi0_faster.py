import functools
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

STANDARD_TRAINING_MODE = "standard"
BLOCK_CAUSAL_FORCING_MODE = "block_causal_forcing"


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def make_block_causal_action_ar_mask(action_horizon: int, block_size: int) -> jax.Array:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return (jnp.arange(action_horizon) % block_size) == 0


def make_block_causal_action_mask(action_horizon: int, block_size: int) -> jax.Array:
    ar_mask = make_block_causal_action_ar_mask(action_horizon, block_size)
    input_mask = jnp.ones((1, action_horizon), dtype=jnp.bool_)
    return make_attn_mask(input_mask, ar_mask)[0]


def sample_block_timesteps(
    rng: at.KeyArrayLike, batch_size: int, action_horizon: int, block_size: int
) -> tuple[jax.Array, jax.Array]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    num_blocks = (action_horizon + block_size - 1) // block_size
    block_time = jax.random.beta(rng, 1.5, 1, (batch_size, num_blocks)) * 0.999 + 0.001
    block_ids = jnp.arange(action_horizon) // block_size
    return block_time[:, block_ids], block_time


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0Faster(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0FasterConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True
        self.max_delay = config.max_delay
        self.training_mode = config.training_mode
        self.block_size = config.block_size
        self.use_block_causal_forcing = self.training_mode == BLOCK_CAUSAL_FORCING_MODE

        self.mix_prob = config.mix_prob
        assert 0.0 <= self.mix_prob <= 1.0, "mix_prob must be in [0, 1]"
        self.alpha = config.alpha
        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0, 1]"
        self.u0 = config.u0
        assert 0.0 <= self.u0 <= 1.0, "u0 must be in [0, 1]"
        print(
            f"mix_prob: {self.mix_prob}, alpha: {self.alpha}, u0: {self.u0}, "
            f"training_mode: {self.training_mode}, block_size: {self.block_size}"
        )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b ah"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b ah emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1], (b, ah, emb)
        time_emb = jax.vmap(
            functools.partial(
                posemb_sincos, embedding_dim=self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
            )
        )(timestep)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            # time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            time_tokens = time_emb
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        if self.use_block_causal_forcing:
            ar_mask += [(i % self.block_size) == 0 for i in range(self.action_horizon)]
        else:
            ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _compute_block_causal_forcing_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_size, action_horizon, _ = actions.shape
        noise = jax.random.normal(noise_rng, actions.shape)
        time, _ = sample_block_timesteps(time_rng, batch_size, action_horizon, self.block_size)
        x_t = time[..., None] * noise + (1 - time[..., None]) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.use_block_causal_forcing:
            return self._compute_block_causal_forcing_loss(rng, observation, actions, train=train)

        preprocess_rng, noise_rng, time_rng, delay_rng, type_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        b, ah, ad = actions.shape
        noise = jax.random.normal(noise_rng, actions.shape)

        # Generate random delay for each batch
        delay = jax.random.randint(delay_rng, (b,), 0, self.max_delay)
        # Create mask where positions < delay are True
        prefix_action_mask = jnp.arange(ah)[None, :] < delay[:, None]  # (b, ah)

        time_const = jax.random.beta(time_rng, 1.5, 1, (b, 1)) * 0.999 + 0.001
        time_const = jnp.broadcast_to(time_const, (b, ah))
        time_HAS = self.compute_HAS(time_const, delay, alpha=self.alpha, u0=self.u0)

        use_HAS = jax.random.bernoulli(type_rng, self.mix_prob, (b, 1))
        time = jnp.where(use_HAS, time_HAS, time_const)

        # Set time to 0 where mask is True (prefix positions get ground truth actions)
        time = jnp.where(prefix_action_mask, 0.0, time)  # (b, ah)

        x_t = time[..., None] * noise + (1 - time[..., None]) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # Only compute loss where prefix_mask is False (i.e., not in the prefix region).
        # prefix_mask is (b, prefix_len), but v_t and u_t are (b, ah, ad), so we need to broadcast and only select the suffix positions.
        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # (b, ah)
        postfix_action_mask = jnp.logical_not(prefix_action_mask)  # (b, ah)
        loss = jnp.sum(loss * postfix_action_mask) / (jnp.sum(postfix_action_mask) + 1e-8)
        return loss

    def compute_HAS(
        self, time: jax.Array, delay: jax.Array | None = None, alpha: float = 1.0, u0: float = 0.9
    ) -> jax.Array:
        """
        Horizon-Aware Schedule
        time: (b, 1) or (a, b, 1)
        delay: (b,)
        return: (b, ah) or (a, b, ah)
        """
        i = jnp.arange(self.action_horizon)[None, :]  # (1, ah)
        i_valid = jnp.maximum(i - delay[:, None], 0)  # (b, ah), can be negative for positions < delay
        denom = jnp.maximum(self.action_horizon - 1 - delay, 1)[:, None]  # (b, 1)

        j = i_valid / denom  # (b, ah)
        u = (1 - j**alpha) * u0  # (b, ah)

        if time.ndim == 3:
            u = u[None, :, :]

        time_schedule = (time - u) / (1 - u)  # (b, ah) or (a, b, ah)

        time_schedule = jnp.clip(time_schedule, 0.0, 1.0)
        return time_schedule

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        delay: at.Int[at.Array, "b"] | None = None,
        action_prefix: at.Float[at.Array, "b ah ad"] | None = None,
        infer_time_schedule: str = "const",
        alpha: float = 1.0,
        u0: float = 0.9,
    ) -> _model.Actions:
        if self.use_block_causal_forcing and infer_time_schedule != "const":
            raise ValueError("block_causal_forcing currently supports only const inference schedule")

        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if delay is None:
            delay = jnp.zeros((batch_size,), dtype=jnp.int32)
            action_prefix = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        assert action_prefix.shape == (batch_size, self.action_horizon, self.action_dim)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        prefix_action_mask = jnp.arange(self.action_horizon)[None, :] < delay[:, None]  # (b, ah)

        def step(carry, _):
            x_t, time = carry
            x_t = jnp.where(prefix_action_mask[..., None], action_prefix, x_t)
            time_ = jnp.broadcast_to(time, batch_size)  # (b, )
            time_ = jnp.where(prefix_action_mask, 0.0, time_[:, None])  # (b, ah)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time_)
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            x_next = x_t + dt * v_t
            return (x_next, time + dt), None

        def step_adaptive(carry, step_params):
            x_t, _ = carry
            x_t = jnp.where(prefix_action_mask[..., None], action_prefix, x_t)
            t_curr, dt_curr = step_params  # (b, ah)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, t_curr)
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            x_next = x_t + dt_curr[..., None] * v_t
            return (x_next, None), None

        if infer_time_schedule == "const":
            (x_0, _), _ = jax.lax.scan(step, (noise, 1.0), None, length=num_steps)
        elif infer_time_schedule == "HAS":
            base_times = jnp.linspace(1.0, 0.0, num_steps + 1)[:, None, None]  # (num_steps + 1, b, 1)

            t_schedule = self.compute_HAS(base_times, delay, alpha=alpha, u0=u0)  # (num_steps + 1, b, ah)
            t_schedule = jnp.where(prefix_action_mask[None, :, :], 0.0, t_schedule)

            dt_schedule = t_schedule[1:] - t_schedule[:-1]
            t_starts = t_schedule[:-1]

            (x_0, _), _ = jax.lax.scan(step_adaptive, (noise, None), (t_starts, dt_schedule), length=num_steps)
        else:
            raise ValueError(f"Invalid infer_time_schedule: {infer_time_schedule}")

        return x_0

    def sample_actions_streaming_init(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        delay: at.Int[at.Array, "b"] | None = None,
        action_prefix: at.Float[at.Array, "b ah ad"] | None = None,
        alpha: float = 1.0,
        u0: float = 0.9,
    ):
        """Precomputes kv_cache and time schedules before streaming."""
        if self.use_block_causal_forcing:
            raise NotImplementedError("block_causal_forcing streaming inference is not implemented yet")

        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if delay is None:
            delay = jnp.zeros((batch_size,), dtype=jnp.int32)
            action_prefix = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        assert action_prefix.shape == (batch_size, self.action_horizon, self.action_dim)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask_init = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask_init, positions=positions)

        prefix_action_mask = jnp.arange(self.action_horizon)[None, :] < delay[:, None]

        time = jnp.linspace(1.0, 0.0, num_steps + 1)[:, None, None]
        t_schedule = self.compute_HAS(time, delay, alpha=alpha, u0=u0)  # (num_steps + 1, b, ah)
        t_schedule = jnp.where(prefix_action_mask[None, :, :], 0.0, t_schedule)

        dt_schedule = t_schedule[1:] - t_schedule[:-1]
        t_starts = t_schedule[:-1]
        is_ready_after_step = t_schedule[1:] < 0.01  # mark actions as ready when time is close to 0

        already_output_init = prefix_action_mask  # clean actions

        return (
            noise,
            already_output_init,
            t_starts,
            dt_schedule,
            is_ready_after_step,
            kv_cache,
            prefix_mask,
            prefix_action_mask,
            action_prefix,
            observation,
        )

    def sample_actions_streaming_step(
        self,
        x_t: jax.Array,
        already_output: jax.Array,
        t_curr: jax.Array,
        dt_curr: jax.Array,
        step_ready: jax.Array,
        kv_cache,
        prefix_mask: jax.Array,
        prefix_action_mask: jax.Array,
        action_prefix: jax.Array,
        observation: _model.Observation,
    ):
        """Single streaming step, designed to be called asynchronously in a host loop."""
        x_t = jnp.where(prefix_action_mask[..., None], action_prefix, x_t)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, t_curr)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        x_next = x_t + dt_curr[..., None] * v_t

        newly_ready = step_ready & ~already_output
        already_output_next = already_output | step_ready

        return x_next, already_output_next, newly_ready
