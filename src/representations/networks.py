from typing import Any, Callable, Dict, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import utils.hk as hku

ModuleBuilder = Callable[[], Callable[[jax.Array | np.ndarray], jax.Array]]


class NetworkBuilder:
    def __init__(self, input_shape: Tuple, params: Dict[str, Any], key: jax.Array):
        self._input_shape = tuple(input_shape)
        self._h_params = params
        self._rng, feat_rng = jax.random.split(key)

        self._feat_net, self._feat_params = buildFeatureNetwork(
            input_shape, params, feat_rng
        )

        self._params = {
            "phi": self._feat_params,
        }

        self._retrieved_params = False

        print(hk.experimental.tabulate(self._feat_net)(np.ones((1,) + self._input_shape)))

    def getParams(self):
        self._retrieved_params = True
        return self._params

    def getFeatureFunction(self):
        def _inner(params: Any, x: jax.Array | np.ndarray):
            return self._feat_net.apply(params["phi"], x)

        return _inner
    
    def getRecurrentFeatureFunction(self):
        def _inner(params: Any, x: jax.Array | np.ndarray, reset: jax.Array | np.ndarray = None, carry: jax.Array | np.ndarray = None, is_target = False):
            return self._feat_net.apply(params['phi'], x, reset=reset, carry=carry, is_target=is_target)

        return _inner
    
    def getMultiplicativeActionRecurrentFeatureFunction(self):
        def _inner(params: Any, x: jax.Array | np.ndarray, a: jax.Array, reset: jax.Array | np.ndarray = None, carry: jax.Array | np.ndarray = None, is_target = False):
            return self._feat_net.apply(params['phi'], x, a, reset=reset, carry=carry, is_target=is_target)

        return _inner

    def addHead(
        self, module: ModuleBuilder, name: Optional[str] = None, grad: bool = True
    ):
        assert not self._retrieved_params, (
            "Attempted to add head after params have been retrieved"
        )
        _state = {}

        def _builder(x: jax.Array | np.ndarray):
            head = module()
            _state["name"] = getattr(head, "name", None)

            if not grad:
                x = jax.lax.stop_gradient(x)

            out = head(x)
            return out

        sample_in = jnp.zeros((1,) + self._input_shape)
        if 'GRU' in self._h_params['type']:
            sample_phi = self._feat_net.apply(self._feat_params, sample_in)[0]
        else:
            sample_phi = self._feat_net.apply(self._feat_params, sample_in).out
        self._sample_phi = sample_phi

        self._rng, rng = jax.random.split(self._rng)
        h_net = hk.without_apply_rng(hk.transform(_builder))
        h_params = h_net.init(rng, sample_phi)
        print(hk.experimental.tabulate(h_net)(sample_phi))

        name = name or _state.get("name")
        assert name is not None, "Could not detect name from module"
        self._params[name] = h_params

        def _inner(params: Any, x: jax.Array):
            return h_net.apply(params[name], x)

        return h_net, h_params, _inner

    def reset(self, key: jax.Array):
        sample_in = jnp.zeros((1,) + self._input_shape)
        return {
            "phi": self._feat_net.init(key, sample_in),
        }

def reluLayers(
    layers: List[int],
    name: Optional[str] = None,
    layer_norm: bool = False,
    w_init: Optional[hk.initializers.Initializer] = None,
    b_init: Optional[hk.initializers.Initializer] = None,
):
    if w_init is None:
        w_init = hk.initializers.Orthogonal(np.sqrt(2))
    if b_init is None:
        b_init = hk.initializers.Constant(0)

    out = []
    for width in layers:
        out.append(hk.Linear(width, w_init=w_init, b_init=b_init, name=name))
        if layer_norm:
            out.append(hk.LayerNorm(axis=-1, create_scale=True, create_offset=True))
        out.append(jax.nn.relu)

    return out


def buildFeatureNetwork(inputs: Tuple, params: Dict[str, Any], rng: Any):
    def _inner(x: jax.Array, *args, **kwargs):
        name = params["type"]
        hidden = params["hidden"]

        if name == "TwoLayerRelu":
            layers = reluLayers([hidden, hidden], name="phi")

        elif name == "OneLayerRelu":
            layers = reluLayers([hidden], name="phi")

        elif name == "MinatarNet":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 2, w_init=w_init, name="phi"),
                jax.nn.relu,
                hk.Flatten(name="phi"),
            ]

        elif name == "ForagerNet":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 1, w_init=w_init, name="phi"),
                jax.nn.relu,
                hk.Flatten(name="phi"),
            ]

        elif name == "ForagerLayerNormNet":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 1, w_init=w_init, name="phi"),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Flatten(name="phi"),
            ]

        elif name == "Forager2Net":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 1, w_init=w_init, name="phi"),
                jax.nn.relu,
                hk.Flatten(name="phi"),
            ]

        elif name == "Forager2LayerNormNet":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 1, w_init=w_init, name="phi"),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Flatten(name="phi"),
            ]
            layers += reluLayers([hidden], name="phi")
            
        elif name == 'ForagerGRUNetReLU':
            # It uses initializer different from above
            net = ForagerGRUNetReLU(hidden=hidden, learn_initial_h=params.get('learn_initial_h', True), name='ForagerGRUNetReLU')
            return net(x, *args, **kwargs)
        
        elif name == 'ForagerGRUNetReLU2':
            # It uses initializer different from above
            net = ForagerGRUNetReLU2(hidden=hidden, learn_initial_h=params.get('learn_initial_h', True), name='ForagerGRUNetReLU2')
            return net(x, *args, **kwargs)
        
        elif name == 'ForagerGRUNetReLU3':
            # It uses initializer different from above
            net = ForagerGRUNetReLU3(hidden=hidden, learn_initial_h=params.get('learn_initial_h', True), name='ForagerGRUNetReLU3')
            return net(x, *args, **kwargs)
        
        elif name == 'ForagerGRUNetReLU3Xavier':
            # It uses initializer different from above
            net = ForagerGRUNetReLU3Xavier(hidden=hidden, learn_initial_h=params.get('learn_initial_h', True), name='ForagerGRUNetReLU3Xavier')
            return net(x, *args, **kwargs)
        
        elif name == 'ForagerMAGRUNetReLU':
            # It uses initializer different from above
            net = ForagerMAGRUNetReLU(hidden=hidden, actions=params["actions"], learn_initial_h=params.get('learn_initial_h', True), name='ForagerMAGRUNetReLU')
            return net(x, *args, **kwargs)
        
        elif name == 'ForagerMAGRUNetReLU2':
            # It uses initializer different from above
            net = ForagerMAGRUNetReLU2(hidden=hidden, actions=params["actions"], learn_initial_h=params.get('learn_initial_h', True), name='ForagerMAGRUNetReLU2')
            return net(x, *args, **kwargs)
        
        elif name == 'ForagerAAGRUNetReLU3Xavier':
            # It uses initializer different from above
            net = ForagerAAGRUNetReLU3Xavier(hidden=hidden, actions=params["actions"], learn_initial_h=params.get('learn_initial_h', True), name='ForagerAAGRUNetReLU3Xavier')
            return net(x, *args, **kwargs)
        
        elif name == 'ForagerMAGRUNetReLU3':
            # It uses initializer different from above
            net = ForagerMAGRUNetReLU3(hidden=hidden, actions=params["actions"], learn_initial_h=params.get('learn_initial_h', True), name='ForagerMAGRUNetReLU3')
            return net(x, *args, **kwargs)

        elif name == "AtariNet":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                lambda x: x.astype(np.float32),
                make_conv(32, (8, 8), (4, 4)),
                jax.nn.relu,
                make_conv(64, (4, 4), (2, 2)),
                jax.nn.relu,
                make_conv(64, (3, 3), (1, 1)),
                jax.nn.relu,
                hk.Flatten(),
                hk.Linear(512, w_init=w_init),
                jax.nn.relu,
            ]

        else:
            raise NotImplementedError()

        return hku.accumulatingSequence(layers)(x)

    network = hk.without_apply_rng(hk.transform(_inner))

    sample_input = jnp.zeros((1,) + tuple(inputs))
    net_params = network.init(rng, sample_input)

    return network, net_params


def make_conv(size: int, shape: Tuple[int, int], stride: Tuple[int, int]):
    w_init = hk.initializers.Orthogonal(np.sqrt(2))
    b_init = hk.initializers.Constant(0)
    return hk.Conv2D(
        size,
        kernel_shape=shape,
        stride=stride,
        w_init=w_init,
        b_init=b_init,
        padding="VALID",
        name="conv",
    )

class GRU(hk.Module):
    def __init__(self, hidden: int, learn_initial_h=True, w_init=hk.initializers.Orthogonal(np.sqrt(2)), name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        self.gru = hk.GRU(self.hidden, name='gru_inner', w_h_init=w_init, w_i_init=w_init)
        self.learn_initial_h = learn_initial_h
        
    def initial_state(self, batch=1, length=1):
        if self.learn_initial_h:
            init_h = hk.get_parameter("initial_h", shape=(self.hidden,), init=jnp.zeros)
            init_h = jnp.repeat(init_h[None, :], batch, axis=0)
            init_h = jnp.repeat(init_h[:, None, :], length, axis=1)
        else:
            # This is all zeros
            init_h = jnp.repeat(self.gru.initial_state(batch_size=batch)[:, None, :], length, axis=1)
        return init_h
        
    def gru_step(self, prev_state, inputs):
        frame_feat, reset_flag, carry = inputs
        # Reset state if flag is True.
        prev_state = jax.lax.select(reset_flag, carry[None, :], prev_state)
        # GRU expects inputs with a batch dimension.
        output, next_state = self.gru(frame_feat[None, :], prev_state)
        # Remove the extra batch dimension and return both output and next_state.
        return next_state, (output[0], next_state[0])

    def process_sequence(self, features_seq, reset_seq, carry_seq):
        final_state, (outputs_seq, state_seq) = hk.scan(self.gru_step, carry_seq[:1, :], (features_seq, reset_seq, carry_seq))
        return outputs_seq, state_seq
    
    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """

        N, T, *_ = x.shape

        if reset is None:
            reset = jnp.zeros((N, T), dtype=bool)

        if carry is None:
            carry = self.initial_state(N, T)
        elif len(carry.shape) < 3:
            carry = carry[:, None, :]
        
        # Replace entries in carry where reset is true with the initial state
        if self.learn_initial_h and not is_target:
            init_state = self.initial_state(N, T)
            carry = jnp.where(reset[..., None], init_state, carry)

        # Vectorize the per-sequence unroll over the batch dimension.
        # x has shape [N, T, ...] and reset has shape [N, T].
        outputs_sequence, states_sequence = jax.vmap(self.process_sequence)(x, reset, carry)

        # Return both the GRU outputs and hidden states across the entire sequence.
        return outputs_sequence, states_sequence, self.initial_state(1, 1)[:, 0, ...]

class MAGRU(hk.Module):
    def __init__(self, hidden: int, actions: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden_size = hidden
        self.number_of_actions = actions
        self.w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.b_init = jnp.zeros
        self.learn_initial_h = learn_initial_h
        
    def initial_state(self, batch=1, length=1):
        if self.learn_initial_h:
            init_h = hk.get_parameter("initial_h", shape=(self.hidden_size,), init=jnp.zeros)
            init_h = jnp.repeat(init_h[None, :], batch, axis=0)
            init_h = jnp.repeat(init_h[:, None, :], length, axis=1)
        else:
            # This is all zeros
            init_h = jnp.broadcast_to(jnp.zeros([self.hidden_size]), (batch, length, self.hidden_size))
        return init_h
    
    def gru_call(self, inputs, action, state):
        # modified from https://github.com/google-deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py#L521#L588
        self.input_size = inputs.shape[-1]
        w_i = hk.get_parameter("w_i", [self.number_of_actions, self.input_size, 3 * self.hidden_size], init=self.w_init)
        w_h = hk.get_parameter("w_h", [self.number_of_actions, self.hidden_size, 3 * self.hidden_size], init=self.w_init)
        b = hk.get_parameter("b", [self.number_of_actions, 3 * self.hidden_size], init=self.b_init)
        w_i = w_i[action]
        w_h = w_h[action]
        b = b[action]
        w_h_z, w_h_a = jnp.split(w_h, indices_or_sections=[2 * self.hidden_size], axis=1)
        b_z, b_a = jnp.split(b, indices_or_sections=[2 * self.hidden_size], axis=0)

        gates_x = jnp.matmul(inputs, w_i)
        zr_x, a_x = jnp.split(
            gates_x, indices_or_sections=[2 * self.hidden_size], axis=-1)
        zr_h = jnp.matmul(state, w_h_z)
        zr = zr_x + zr_h + jnp.broadcast_to(b_z, zr_h.shape)
        z, r = jnp.split(jax.nn.sigmoid(zr), indices_or_sections=2, axis=-1)

        a_h = jnp.matmul(r * state, w_h_a)
        a = jnp.tanh(a_x + a_h + jnp.broadcast_to(b_a, a_h.shape))

        next_state = (1 - z) * state + z * a
        return next_state, next_state
        
    def gru_step(self, prev_state, inputs):
        frame_feat, action, reset_flag, carry = inputs
        # Reset state if flag is True.
        prev_state = jax.lax.select(reset_flag, carry[None, :], prev_state)
        # GRU expects inputs with a batch dimension.
        output, next_state = self.gru_call(frame_feat[None, :], action, prev_state)
        # Remove the extra batch dimension and return both output and next_state.
        return next_state, (output[0], next_state[0])

    def process_sequence(self, features_seq, action_seq, reset_seq, carry_seq):
        final_state, (outputs_seq, state_seq) = hk.scan(self.gru_step, carry_seq[:1, :], (features_seq, action_seq, reset_seq, carry_seq))
        return outputs_seq, state_seq
    
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          a: Action tensor with shape [N, T]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """

        N, T, *_ = x.shape

        if reset is None:
            reset = jnp.zeros((N, T), dtype=bool)

        if carry is None:
            carry = self.initial_state(N, T)
        elif len(carry.shape) < 3:
            carry = carry[:, None, :]

        # Replace entries in carry where reset is true with the initial state
        if self.learn_initial_h and not is_target:
            init_state = self.initial_state(N, T)
            carry = jnp.where(reset[..., None], init_state, carry)

        # Vectorize the per-sequence unroll over the batch dimension.
        # x has shape [N, T, ...], a has shape [N, T], and reset has shape [N, T].
        outputs_sequence, states_sequence = jax.vmap(self.process_sequence)(x, a, reset, carry)

        # Return both the GRU outputs and hidden states across the entire sequence.
        return outputs_sequence, states_sequence, self.initial_state(1, 1)[:, 0, ...]

class AAGRU(hk.Module):
    def __init__(self, hidden: int, actions: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden_size = hidden
        self.number_of_actions = actions
        self.w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.b_init = jnp.zeros
        self.learn_initial_h = learn_initial_h
        
    def initial_state(self, batch=1, length=1):
        if self.learn_initial_h:
            init_h = hk.get_parameter("initial_h", shape=(self.hidden_size,), init=jnp.zeros)
            init_h = jnp.repeat(init_h[None, :], batch, axis=0)
            init_h = jnp.repeat(init_h[:, None, :], length, axis=1)
        else:
            # This is all zeros
            init_h = jnp.broadcast_to(jnp.zeros([self.hidden_size]), (batch, length, self.hidden_size))
        return init_h
    
    def gru_call(self, inputs, action, state):
        # modified from https://github.com/google-deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py#L521#L588
        self.input_size = inputs.shape[-1]
        w_i = hk.get_parameter("w_i", [self.input_size, 3 * self.hidden_size], init=self.w_init)
        w_h = hk.get_parameter("w_h", [self.hidden_size, 3 * self.hidden_size], init=self.w_init)
        w_a = hk.get_parameter("w_a", [self.number_of_actions, 3 * self.hidden_size], init=self.w_init)
        b = hk.get_parameter("b", [3 * self.hidden_size], init=self.b_init)

        w_h_z, w_h_a = jnp.split(w_h, indices_or_sections=[2 * self.hidden_size], axis=1)
        b_z, b_a = jnp.split(b, indices_or_sections=[2 * self.hidden_size], axis=0)

        gates_x = jnp.matmul(inputs, w_i)
        gates_a = jnp.matmul(action, w_a) 
        
        zr_x, a_x = jnp.split(gates_x, indices_or_sections=[2 * self.hidden_size], axis=-1)
        zr_a, a_a = jnp.split(gates_a, indices_or_sections=[2 * self.hidden_size], axis=-1)
        
        zr_h = jnp.matmul(state, w_h_z)
        
        zr = zr_x + zr_h + zr_a + jnp.broadcast_to(b_z, zr_h.shape)
        z, r = jnp.split(jax.nn.sigmoid(zr), indices_or_sections=2, axis=-1)

        a_h = jnp.matmul(r * state, w_h_a)
        a = jnp.tanh(a_x + a_h + a_a + jnp.broadcast_to(b_a, a_h.shape))

        next_state = (1 - z) * state + z * a
        return next_state, next_state
        
    def gru_step(self, prev_state, inputs):
        frame_feat, action, reset_flag, carry = inputs
        # Reset state if flag is True.
        prev_state = jax.lax.select(reset_flag, carry[None, :], prev_state)
        # GRU expects inputs with a batch dimension.
        output, next_state = self.gru_call(frame_feat[None, :], action, prev_state)
        # Remove the extra batch dimension and return both output and next_state.
        return next_state, (output[0], next_state[0])

    def process_sequence(self, features_seq, action_seq, reset_seq, carry_seq):
        final_state, (outputs_seq, state_seq) = hk.scan(self.gru_step, carry_seq[:1, :], (features_seq, action_seq, reset_seq, carry_seq))
        return outputs_seq, state_seq
    
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          a: Action tensor with shape [N, T]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """

        N, T, *_ = x.shape

        if reset is None:
            reset = jnp.zeros((N, T), dtype=bool)

        if carry is None:
            carry = self.initial_state(N, T)
        elif len(carry.shape) < 3:
            carry = carry[:, None, :]

        # Replace entries in carry where reset is true with the initial state
        if self.learn_initial_h and not is_target:
            init_state = self.initial_state(N, T)
            carry = jnp.where(reset[..., None], init_state, carry)

        # Vectorize the per-sequence unroll over the batch dimension.
        # x has shape [N, T, ...], a has shape [N, T, num_action], and reset has shape [N, T].
        outputs_sequence, states_sequence = jax.vmap(self.process_sequence)(x, a, reset, carry)

        # Return both the GRU outputs and hidden states across the entire sequence.
        return outputs_sequence, states_sequence, self.initial_state(1, 1)[:, 0, ...]

class ForagerGRUNetReLU(hk.Module):
    def __init__(self, hidden: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        self.conv = hk.Conv2D(16, 3, 2, w_init=w_init, name='phi')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')
        
        self.skip_connection = hk.Linear(self.hidden, w_init=w_init, name='skip_connection')

        self.gru = GRU(self.hidden, learn_initial_h=learn_initial_h, name='gru')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
        
        N, T, *feat = x.shape
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
        
        outputs_sequence, states_sequence, initial_carry = self.gru(h, reset, carry, is_target=is_target)
        
        outputs_sequence = jax.nn.relu(outputs_sequence)
        
        outputs_sequence = outputs_sequence + self.skip_connection(h)
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry
    
class ForagerGRUNetReLU2(hk.Module):
    def __init__(self, hidden: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        self.conv = hk.Conv2D(16, 3, 2, w_init=w_init, name='phi')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')
        
        self.skip_connection = hk.Linear(self.hidden, w_init=w_init, name='skip_connection')

        self.gru = GRU(self.hidden, learn_initial_h=learn_initial_h, name='gru')
        
        self.linear = hk.Linear(self.hidden, w_init=w_init, name='linear')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
        
        N, T, *feat = x.shape
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
        
        outputs_sequence, states_sequence, initial_carry = self.gru(h, reset, carry, is_target=is_target)
        
        outputs_sequence = jax.nn.relu(outputs_sequence)
        
        outputs_sequence = outputs_sequence + self.skip_connection(h)
        
        outputs_sequence = jax.nn.relu(self.linear(outputs_sequence))
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry

class ForagerGRUNetReLU3(hk.Module):
    def __init__(self, hidden: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        self.conv = hk.Conv2D(16, 3, 2, w_init=w_init, name='phi')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')
        
        self.skip_connection = hk.Linear(self.hidden, w_init=w_init, name='skip_connection')

        self.gru = GRU(self.hidden, learn_initial_h=learn_initial_h, name='gru')
        
        self.linear1 = hk.Linear(self.hidden, w_init=w_init, name='linear1')
        self.linear2 = hk.Linear(self.hidden, w_init=w_init, name='linear2')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
        
        N, T, *feat = x.shape
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
        
        outputs_sequence, states_sequence, initial_carry = self.gru(h, reset, carry, is_target=is_target)
        
        outputs_sequence = jax.nn.relu(outputs_sequence)
        
        outputs_sequence = outputs_sequence + self.skip_connection(h)
        
        outputs_sequence = jax.nn.relu(self.linear2(jax.nn.relu(self.linear1(outputs_sequence))))
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry

class ForagerGRUNetReLU3Xavier(hk.Module):
    def __init__(self, hidden: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")

        self.conv = hk.Conv2D(16, 3, 2, w_init=w_init, name='phi')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')
        
        self.skip_connection = hk.Linear(self.hidden, w_init=w_init, name='skip_connection')

        self.gru = GRU(self.hidden, learn_initial_h=learn_initial_h, w_init=w_init, name='gru')
        
        self.linear1 = hk.Linear(self.hidden, w_init=w_init, name='linear1')
        self.linear2 = hk.Linear(self.hidden, w_init=w_init, name='linear2')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
        
        N, T, *feat = x.shape
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
        
        outputs_sequence, states_sequence, initial_carry = self.gru(h, reset, carry, is_target=is_target)
        
        outputs_sequence = jax.nn.relu(outputs_sequence)
        
        # outputs_sequence = outputs_sequence + self.skip_connection(h)
        
        outputs_sequence = jax.nn.relu(self.linear2(jax.nn.relu(self.linear1(outputs_sequence))))
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry

class ForagerAAGRUNetReLU3Xavier(hk.Module):
    def __init__(self, hidden: int, actions: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        self.number_of_actions = actions
        w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")

        self.conv = hk.Conv2D(16, 3, 2, w_init=w_init, name='phi')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')
        
        self.skip_connection = hk.Linear(self.hidden, w_init=w_init, name='skip_connection')

        self.aagru = AAGRU(self.hidden, self.number_of_actions, learn_initial_h=learn_initial_h, name='aagru')
        
        self.linear1 = hk.Linear(self.hidden, w_init=w_init, name='linear1')
        self.linear2 = hk.Linear(self.hidden, w_init=w_init, name='linear2')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, a: jnp.ndarray = None, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
        
        N, T, *feat = x.shape
        
        # Use No-Op action 0 to populate a if None
        if a is None:
            a = jnp.full((N, T, self.number_of_actions), jnp.float32(0))
        if (len(a.shape) < 3):
            a = jnp.broadcast_to(a, (N, T, self.number_of_actions))
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
        
        outputs_sequence, states_sequence, initial_carry = self.aagru(h, a, reset, carry, is_target=is_target)
        
        outputs_sequence = jax.nn.relu(outputs_sequence)
        
        # outputs_sequence = outputs_sequence + self.skip_connection(h)
        
        outputs_sequence = jax.nn.relu(self.linear2(jax.nn.relu(self.linear1(outputs_sequence))))
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry

class ForagerMAGRUNetReLU(hk.Module):
    def __init__(self, hidden: int, actions: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden_size = hidden
        self.number_of_actions = actions
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        self.conv = hk.Conv2D(16, 3, 2, w_init=w_init, name='phi')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')
        
        self.skip_connection = hk.Linear(self.hidden_size, w_init=w_init, name='skip_connection')

        self.magru = MAGRU(self.hidden_size, self.number_of_actions, learn_initial_h=learn_initial_h, name='gru')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, a: jnp.ndarray = None, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          a: Action tensor with shape [N, T]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
            
        N, T, *feat = x.shape

        # Use No-Op action -1 to populate a if None
        if a is None:
            a = jnp.full((N, T), jnp.int32(-1))
        if (len(a.shape) < 2):
            a = jnp.broadcast_to(a, (N, T))
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
        
        outputs_sequence, states_sequence, initial_carry = self.magru(h, a, reset, carry, is_target=is_target)
        
        outputs_sequence = jax.nn.relu(outputs_sequence)
        
        outputs_sequence = outputs_sequence + self.skip_connection(h)
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry
    
class ForagerMAGRUNetReLU2(hk.Module):
    def __init__(self, hidden: int, actions: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden_size = hidden
        self.number_of_actions = actions
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        self.conv = hk.Conv2D(16, 3, 2, w_init=w_init, name='phi')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')
        
        self.skip_connection = hk.Linear(self.hidden_size, w_init=w_init, name='skip_connection')

        self.magru = MAGRU(self.hidden_size, self.number_of_actions, learn_initial_h=learn_initial_h, name='gru')
        
        self.linear =  hk.Linear(self.hidden_size, w_init=w_init, name='linear')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, a: jnp.ndarray = None, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          a: Action tensor with shape [N, T]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
            
        N, T, *feat = x.shape

        # Use No-Op action -1 to populate a if None
        if a is None:
            a = jnp.full((N, T), jnp.int32(-1))
        if (len(a.shape) < 2):
            a = jnp.broadcast_to(a, (N, T))
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
        
        outputs_sequence, states_sequence, initial_carry = self.magru(h, a, reset, carry, is_target=is_target)
        
        outputs_sequence = jax.nn.relu(outputs_sequence)
        
        outputs_sequence = outputs_sequence + self.skip_connection(h)
        
        outputs_sequence = jax.nn.relu(self.linear(outputs_sequence))
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry
    
class ForagerMAGRUNetReLU3(hk.Module):
    def __init__(self, hidden: int, actions: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden_size = hidden
        self.number_of_actions = actions
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        self.conv = hk.Conv2D(16, 3, 2, w_init=w_init, name='phi')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')
        
        self.skip_connection = hk.Linear(self.hidden_size, w_init=w_init, name='skip_connection')

        self.magru = MAGRU(self.hidden_size, self.number_of_actions, learn_initial_h=learn_initial_h, name='gru')
        
        self.linear1 =  hk.Linear(self.hidden_size, w_init=w_init, name='linear1')
        self.linear2 =  hk.Linear(self.hidden_size, w_init=w_init, name='linear2')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, a: jnp.ndarray = None, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          a: Action tensor with shape [N, T]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
            
        N, T, *feat = x.shape

        # Use No-Op action -1 to populate a if None
        if a is None:
            a = jnp.full((N, T), jnp.int32(-1))
        if (len(a.shape) < 2):
            a = jnp.broadcast_to(a, (N, T))
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
        
        outputs_sequence, states_sequence, initial_carry = self.magru(h, a, reset, carry, is_target=is_target)
        
        outputs_sequence = jax.nn.relu(outputs_sequence)
        
        outputs_sequence = outputs_sequence + self.skip_connection(h)
        
        outputs_sequence = jax.nn.relu(self.linear2(jax.nn.relu(self.linear1(outputs_sequence))))
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry