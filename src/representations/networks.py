from typing import Any, Callable, Dict, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import utils.hk as hku

ModuleBuilder = Callable[[], Callable[[jax.Array | np.ndarray], jax.Array]]


class NetworkBuilder:
    def __init__(self, input_shape: Tuple, params: Dict[str, Any], seed: int):
        self._input_shape = tuple(input_shape)
        self._h_params = params
        self._rng, feat_rng = jax.random.split(jax.random.key(seed))

        self._feat_net, self._feat_params = buildFeatureNetwork(
            input_shape, params, feat_rng
        )

        self._params = {
            "phi": self._feat_params,
        }

        self._retrieved_params = False

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

        self._rng, rng = jax.random.split(self._rng)
        h_net = hk.without_apply_rng(hk.transform(_builder))
        h_params = h_net.init(rng, sample_phi)

        name = name or _state.get("name")
        assert name is not None, "Could not detect name from module"
        self._params[name] = h_params

        def _inner(params: Any, x: jax.Array):
            return h_net.apply(params[name], x)

        return _inner


def reluLayers(layers: List[int], name: Optional[str] = None):
    w_init = hk.initializers.Orthogonal(np.sqrt(2))
    b_init = hk.initializers.Constant(0)

    out = []
    for width in layers:
        out.append(hk.Linear(width, w_init=w_init, b_init=b_init, name=name))
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
            layers += reluLayers([hidden], name="phi")

        elif name == "ForagerNet":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 1, w_init=w_init, name="phi"),
                jax.nn.relu,
                hk.Flatten(name="phi"),
            ]
            layers += reluLayers([hidden], name="phi")
            
        elif name == 'ForagerGRUNetReLU':
            # It uses initializer different from above
            net = ForagerGRUNetReLU(hidden=hidden, learn_initial_h=params.get('learn_initial_h', True), name='ForagerGRUNetReLU')
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
    def __init__(self, hidden: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        w_init = hk.initializers.Orthogonal(np.sqrt(2))
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

class ForagerGRUNetReLU(hk.Module):
    def __init__(self, hidden: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        self.conv = hk.Conv2D(16, 3, 2, w_init=w_init, name='phi')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')

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
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry