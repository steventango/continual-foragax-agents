# Modified from esraaelelimy/continuing_ppo
import flax.linen as nn 
import jax
from typing import Optional, Tuple, Union, Any, Sequence, Dict
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp
import numpy as np
import distrax
import functools
from algorithms.nn.rtus.rtus import *


def sparse_init(sparsity=0.95, spectral_radius=0.99, dtype=jnp.float32):
    """
    Returns a Flax-compatible sparse initializer with controlled spectral radius.

    Args:
      sparsity: Fraction of weights that are zero (e.g., 0.95 = 95% zeros)
      spectral_radius: Desired spectral radius of the initialized matrix
      dtype: JAX dtype
    """
    def init(key, shape, dtype=dtype):
        k1, k2 = jax.random.split(key)
        # Sparse uniform weights
        W = jax.random.uniform(k1, shape, minval=-1.0, maxval=1.0, dtype=dtype)
        mask = jax.random.bernoulli(k2, p=1.0 - sparsity, shape=shape)
        W = W * mask
        # Scale weights to match target spectral radius
        eigenvalues = jnp.linalg.eigvals(W)
        sigma = jnp.max(jnp.abs(eigenvalues))
        W = (W / (sigma + 1e-12)) * spectral_radius
        return W
    return init

class ESMAC(nn.Module):
    action_dim: int
    d_hidden: int = 16384
    hidden_size: int = 64
    activation: str = "tanh"
    cont: bool = False
    use_sinusoidal_encoding: bool = False
    use_reward_trace: bool = False
    @nn.compact
    def __call__(self, hidden, obs):
        '''
        hidden: ((batch_size, d_hidden), (batch_size, d_hidden))
        obs: ((batch_size, obs_dim), (batch_size, action_dim), (batch_size, 1))
        '''
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        (obs, last_action_encoded, last_reward, sine, cosine, reward_trace) = obs
        last_reward_plus = last_reward
        if self.use_sinusoidal_encoding:
            last_reward_plus = jnp.concatenate((last_reward_plus, sine, cosine), axis=-1)
        if self.use_reward_trace:
            last_reward_plus = jnp.concatenate((last_reward_plus, reward_trace), axis=-1)

        obs = jnp.reshape(obs, (obs.shape[0], -1))
        obs = jnp.concatenate((obs, last_action_encoded, last_reward_plus), axis=-1)
        
        input_embedding = nn.Dense(self.hidden_size,
                                   kernel_init=orthogonal(np.sqrt(2)),
                                   use_bias=False, name="frozen_input_embedding_encoder")(obs)
        input_embedding = nn.Dense(self.d_hidden,
                                   kernel_init=orthogonal(np.sqrt(2)),
                                   use_bias=False, name="frozen_input_embedding_decoder")(input_embedding)
        hidden_embedding = nn.Dense(self.hidden_size,
                                    kernel_init=orthogonal(np.sqrt(2)),
                                    # kernel_init=sparse_init(sparsity=0.95, spectral_radius=0.99),
                                    use_bias=False, name="frozen_hidden_embedding_encoder")(hidden)
        hidden_embedding = nn.Dense(self.d_hidden,
                                    kernel_init=orthogonal(np.sqrt(2)),
                                    # kernel_init=sparse_init(sparsity=0.95, spectral_radius=0.99),
                                    use_bias=False, name="frozen_hidden_embedding_decoder")(hidden_embedding)
        
        state_embedding = jax.lax.stop_gradient(activation(input_embedding + hidden_embedding))

        actor_dense = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor_dense")(state_embedding)
        actor_dense = activation(actor_dense)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_mean")(actor_dense)
        #actor_mean: (batch_size, action_dim)
        if self.cont:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic_dense = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic_dense")(state_embedding)
        critic_dense = activation(critic_dense)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value")(critic_dense)
        #critic: (batch_size, 1)
        hidden = state_embedding
        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_memory(batch_size, d_hidden, d_input):
        carry_init = jnp.zeros((batch_size,d_hidden))
        return carry_init