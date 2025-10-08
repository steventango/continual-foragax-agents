# Modified from esraaelelimy/continuing_ppo
import flax.linen as nn 
from typing import Optional, Tuple, Union, Any, Sequence, Dict
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp
import numpy as np
import distrax
import functools
from algorithms.nn.rtus.rtus import *


class RealTimeActorCriticConvEmb(nn.Module):
    action_dim: Sequence[int]
    d_hidden: int = 192
    hidden_size: int = 64
    activation: str = "tanh"
    cont: bool = False
    rtu_type: str = 'linear_rtu'
    @nn.compact
    def __call__(self, hidden, obs):
        '''
        hidden: (batch_size, d_hidden)
        obs: (seq_len, batch_size, obs_dim)
        '''
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
            
        if self.rtu_type == 'linear_rtu':
            seq_model = RTLRTUs
        elif self.rtu_type == 'non_linear_rtu':
            seq_model = RTNLRTUs    
        else:
            raise NotImplementedError
    
        (actor_hidden, critic_hidden) = hidden

        (obs, last_action_encoded, last_reward) = obs
        obs_hidden_size = self.hidden_size - last_action_encoded.shape[-1] - last_reward.shape[-1]
        
        actor_embedding = nn.Conv(16, 3, 1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_conv")(obs)
        actor_embedding = activation(actor_embedding)
        actor_embedding = jnp.reshape(actor_embedding, (actor_embedding.shape[0], -1))
        actor_embedding = nn.Dense(obs_hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_dense1")(actor_embedding)
        actor_embedding = activation(actor_embedding)
        actor_embedding = jnp.append(actor_embedding, jnp.append(last_action_encoded, last_reward, axis=1), axis=1)
        actor_embedding_skip = actor_embedding
        
        critic_embedding = nn.Conv(16, 3, 1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_conv")(obs)
        critic_embedding = activation(critic_embedding)
        critic_embedding = jnp.reshape(critic_embedding, (critic_embedding.shape[0], -1))
        critic_embedding = nn.Dense(obs_hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_dense1")(critic_embedding)
        critic_embedding = activation(critic_embedding)
        critic_embedding = jnp.append(critic_embedding, jnp.append(last_action_encoded, last_reward, axis=1), axis=1)
        critic_embedding_skip = critic_embedding
        
        actor_hidden, actor_embedding = seq_model(self.d_hidden,params_type='exp_exp', name="actor_rtu")(actor_hidden, actor_embedding)
        critic_hidden, critic_embedding = seq_model(self.d_hidden,params_type='exp_exp', name="critic_rtu")(critic_hidden, critic_embedding)
        actor_embedding = jnp.append(actor_embedding, actor_embedding_skip, axis=1)
        critic_embedding = jnp.append(critic_embedding, critic_embedding_skip, axis=1)
        
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor_dense2")(actor_embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_mean")(actor_mean)
        #actor_mean: (seq_len, batch_size, action_dim)
        if self.cont:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic_dense2")(critic_embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value")(critic)
        #critic: (seq_len, batch_size, 1)
        hidden = (actor_hidden, critic_hidden)
        return hidden, pi, jnp.squeeze(critic, axis=-1)
    @staticmethod
    def initialize_memory(batch_size, d_hidden, d_input):
        actor_hidden_init = (jnp.zeros((batch_size,d_hidden)),jnp.zeros((batch_size,d_hidden)))
        actor_memory_grad_init = (jnp.zeros((batch_size,d_hidden)),jnp.zeros((batch_size,d_hidden)),
                            jnp.zeros((batch_size,d_hidden)),jnp.zeros((batch_size,d_hidden)),
                            jnp.zeros((batch_size,d_input, d_hidden)),jnp.zeros((batch_size,d_input, d_hidden)),
                            jnp.zeros((batch_size,d_input, d_hidden)),jnp.zeros((batch_size,d_input, d_hidden)))
        
        critic_hidden_init = (jnp.zeros((batch_size,d_hidden)),jnp.zeros((batch_size,d_hidden)))
        critic_memory_grad_init = (jnp.zeros((batch_size,d_hidden)),jnp.zeros((batch_size,d_hidden)),
                            jnp.zeros((batch_size,d_hidden)),jnp.zeros((batch_size,d_hidden)),
                            jnp.zeros((batch_size,d_input, d_hidden)),jnp.zeros((batch_size,d_input, d_hidden)),
                            jnp.zeros((batch_size,d_input, d_hidden)),jnp.zeros((batch_size,d_input, d_hidden)))
        
        carry_init = ((actor_hidden_init,actor_memory_grad_init),(critic_hidden_init, critic_memory_grad_init))
        return carry_init