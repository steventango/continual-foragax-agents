import functools
import logging
from typing import Dict, Tuple

import haiku as hk
import jax
from jax import random
import jax.numpy as jnp
import optax


def leastk_mask(scores, ones_fraction):
    if ones_fraction is None or ones_fraction == 0:
        return jnp.zeros_like(scores)
    # This is to ensure indices with smallest values are selected.
    scores = -scores

    n_ones = jnp.round(jnp.size(scores) * ones_fraction)
    # Ensure k is at least 1 and at most the size of scores
    k = jnp.clip(n_ones, 1, jnp.size(scores)).astype(int)

    flat_scores = jnp.reshape(scores, -1)
    sorted_scores = jax.lax.sort(flat_scores)
    # Use dynamic index with jnp.take or similar if needed, but sorted_scores[-k] might work in simple JIT
    # For JIT safety with dynamic k, we can use a more robust approach:
    threshold = sorted_scores[jnp.size(scores) - k]

    mask = (flat_scores >= threshold).astype(flat_scores.dtype)
    return mask.reshape(scores.shape)


def reset_momentum(momentum, mask):
    new_momentum = momentum if mask is None else momentum * (1.0 - mask)
    return new_momentum


def weight_reinit_zero(param, mask):
    if mask is None:
        return param
    else:
        new_param = jnp.zeros_like(param)
        param = jnp.where(mask == 1, new_param, param)
        return param


def weight_reinit_random(
    param, mask, key, weight_scaling=False, scale=1.0, weights_type="incoming"
):
    if mask is None or key is None:
        return param

    # Initialize with xavier_uniform (VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
    # Haiku initializers don't implicitly take PRNGKey as their first argument if they aren't wrapped.
    # Actually, we can just use jax.random.uniform or normal. xavier_uniform is fan_avg + uniform.
    limit = (
        jnp.sqrt(6.0 / (param.shape[-2] + param.shape[-1])) if param.ndim >= 2 else 0.01
    )  # Simple approximations
    if param.ndim == 2:
        limit = jnp.sqrt(6.0 / (param.shape[0] + param.shape[1]))
    elif param.ndim == 4:
        receptive_field_size = param.shape[0] * param.shape[1]
        fan_in = param.shape[2] * receptive_field_size
        fan_out = param.shape[3] * receptive_field_size
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))

    new_param = random.uniform(key, shape=param.shape, minval=-limit, maxval=limit)

    if weight_scaling:
        axes = list(range(param.ndim))
        if weights_type == "outgoing":
            del axes[-2]
        else:
            del axes[-1]

        neuron_mask = jnp.mean(mask, axis=axes)

        non_dead_count = neuron_mask.shape[0] - jnp.count_nonzero(neuron_mask)
        norm_per_neuron = _get_norm_per_neuron(param, axes)
        non_recycled_norm = jnp.sum(norm_per_neuron * (1 - neuron_mask)) / (
            non_dead_count + 1e-9
        )
        non_recycled_norm = non_recycled_norm * scale

        normalized_new_param = _weight_normalization_per_neuron_norm(new_param, axes)
        new_param = normalized_new_param * non_recycled_norm

    param = jnp.where(mask == 1, new_param, param)
    return param


def _weight_normalization_per_neuron_norm(param, axes):
    norm_per_neuron = _get_norm_per_neuron(param, axes)
    norm_per_neuron = jnp.expand_dims(norm_per_neuron, axis=axes)
    normalized_param = param / (norm_per_neuron + 1e-9)
    return normalized_param


def _get_norm_per_neuron(param, axes):
    return jnp.sqrt(jnp.sum(jnp.power(param, 2), axis=axes))


def get_flattened_dict(d, sep="/"):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            sub_dict = get_flattened_dict(v, sep)
            for sub_k, sub_v in sub_dict.items():
                flat_dict[f"{k}{sep}{sub_k}"] = sub_v
        else:
            flat_dict[k] = v
    return flat_dict


def unflatten_dict(flat_dict, sep="/"):
    d = {}
    for k, v in flat_dict.items():
        parts = k.split(sep)
        current = d
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = v
    return d


class BaseRecycler:
    def __init__(
        self,
        all_layers_names,
        dead_neurons_threshold=0.0,
        reset_start_layer_idx=0,
        reset_period=200_000,
        reset_start_step=0,
        reset_end_step=100_000_000,
        logging_period=20_000,
        sub_mean_score=False,
    ):
        self.all_layers_names = all_layers_names
        self.dead_neurons_threshold = dead_neurons_threshold
        self.reset_layers = all_layers_names[reset_start_layer_idx:]
        self.reset_period = reset_period
        self.reset_start_step = reset_start_step
        self.reset_end_step = reset_end_step
        self.logging_period = logging_period
        self.prev_neuron_score = None
        self.sub_mean_score = sub_mean_score

    def update_reset_layers(self, reset_start_layer_idx):
        self.reset_layers = self.all_layers_names[reset_start_layer_idx:]

    def is_update_iter(self, step):
        return step > 0 and (step % self.reset_period == 0)

    def update_weights(self, intermediates, params, key, opt_state):
        raise NotImplementedError

    def maybe_update_weights(self, update_step, intermediates, params, key, opt_state):
        self._last_update_step = update_step
        if self.is_reset(update_step):
            new_params, new_opt_state = self.update_weights(
                intermediates, params, key, opt_state
            )
        else:
            new_params, new_opt_state = params, opt_state
        return new_params, new_opt_state

    def is_reset(self, update_step):
        del update_step
        return False

    def is_intermediated_required(self, update_step):
        return self.is_logging_step(update_step)

    def is_logging_step(self, step):
        return step % self.logging_period == 0

    def maybe_log_deadneurons(self, update_step, intermediates):
        is_logging = self.is_logging_step(update_step)
        if is_logging:
            return self.log_dead_neurons_count(intermediates)
        else:
            return None

    def intersected_dead_neurons_with_last_reset(self, intermediates, update_step):
        if self.is_logging_step(update_step):
            log_dict = self.log_intersected_dead_neurons(intermediates)
            return log_dict
        else:
            return None

    def log_intersected_dead_neurons(self, intermediates):
        score_tree = jax.tree.map(self.estimate_neuron_score, intermediates)
        neuron_score_dict = get_flattened_dict(score_tree, sep="/")

        if self.prev_neuron_score is None:
            self.prev_neuron_score = neuron_score_dict
            log_dict = None
        else:
            log_dict = {}
            for prev_k_score, current_k_score in zip(
                self.prev_neuron_score.items(), neuron_score_dict.items()
            ):
                _, prev_score = prev_k_score
                k, score = current_k_score
                prev_score, score = prev_score[0], score[0]
                prev_mask = prev_score <= self.dead_neurons_threshold
                intersected_mask = (prev_mask) & (score <= self.dead_neurons_threshold)
                prev_dead_count = jnp.count_nonzero(prev_mask)
                intersected_count = jnp.count_nonzero(intersected_mask)

                percent = (
                    (float(intersected_count) / prev_dead_count)
                    if prev_dead_count
                    else 0.0
                )
                log_dict[f"dead_intersected_percent/{k}"] = float(percent) * 100.0

                nondead_mask = score > self.dead_neurons_threshold
                log_dict[f"mean_score_recycled/{k}"] = float(jnp.mean(score[prev_mask]))
                log_dict[f"mean_score_nondead/{k}"] = float(
                    jnp.mean(score[nondead_mask])
                )

            self.prev_neuron_score = neuron_score_dict
        return log_dict

    def log_dead_neurons_count(self, intermediates):
        def log_dict(score, score_type):
            total_neurons, total_deadneurons = 0.0, 0.0
            score_dict = get_flattened_dict(score, sep="/")

            log_dict = {}
            for k, m in score_dict.items():
                if "final_layer" in k or k not in self.reset_layers:
                    continue
                layer_size = float(jnp.size(m))
                deadneurons_count = jnp.count_nonzero(m <= self.dead_neurons_threshold)
                total_neurons += layer_size
                total_deadneurons += deadneurons_count
                log_dict[f"dead_{score_type}_percentage/{k}"] = (
                    float(deadneurons_count) / layer_size
                ) * 100.0
                log_dict[f"dead_{score_type}_count/{k}"] = float(deadneurons_count)
            log_dict[f"{score_type}/total"] = total_neurons
            log_dict[f"{score_type}/deadcount"] = float(total_deadneurons)
            log_dict[f"dead_{score_type}_percentage"] = (
                (float(total_deadneurons) / total_neurons) * 100.0
                if total_neurons > 0
                else 0.0
            )
            return log_dict

        neuron_score = jax.tree.map(self.estimate_neuron_score, intermediates)
        log_dict_neurons = log_dict(neuron_score, "feature")

        return log_dict_neurons

    def estimate_neuron_score(self, activation, is_cbp=False):
        reduce_axes = list(range(activation.ndim - 1))
        if self.sub_mean_score or is_cbp:
            activation = activation - jnp.mean(activation, axis=reduce_axes)

        score = jnp.mean(jnp.abs(activation), axis=reduce_axes)
        if not is_cbp:
            score /= jnp.mean(score) + 1e-9

        return score


class NeuronRecycler(BaseRecycler):
    def __init__(
        self,
        all_layers_names,
        init_method_outgoing="zero",
        weight_scaling=False,
        incoming_scale=1.0,
        outgoing_scale=1.0,
        network="nature",
        **kwargs,
    ):
        super(NeuronRecycler, self).__init__(all_layers_names, **kwargs)
        self.init_method_outgoing = init_method_outgoing
        self.weight_scaling = weight_scaling
        self.incoming_scale = incoming_scale
        self.outgoing_scale = outgoing_scale

        self.next_layers = {}
        for current_layer, next_layer in zip(
            all_layers_names[:-1], all_layers_names[1:]
        ):
            self.next_layers[current_layer] = next_layer

        self.reset_layers = self.reset_layers[:-1]

    def intersected_dead_neurons_with_last_reset(self, intermediates, update_step):
        if self.is_reset(update_step):
            log_dict = self.log_intersected_dead_neurons(intermediates)
            return log_dict
        else:
            return None

    def is_reset(self, update_step):
        within_reset_interval = (
            update_step >= self.reset_start_step and update_step < self.reset_end_step
        )
        return self.is_update_iter(update_step) and within_reset_interval

    def is_intermediated_required(self, update_step):
        is_logging = self.is_logging_step(update_step)
        is_update_iter = self.is_update_iter(update_step)
        return is_logging or is_update_iter

    def update_reset_layers(self, reset_start_layer_idx):
        self.reset_layers = self.all_layers_names[reset_start_layer_idx:]
        self.reset_layers = self.reset_layers[:-1]

    def update_weights(self, intermediates, params, key, opt_state):
        new_param, opt_state = self.recycle_dead_neurons(
            intermediates, params, key, opt_state
        )
        return new_param, opt_state

    def recycle_dead_neurons(self, intermedieates, params, key, opt_state):
        activations_score_dict = get_flattened_dict(intermedieates, sep="/")
        param_dict = get_flattened_dict(params, sep="/")

        (
            incoming_mask_dict,
            outgoing_mask_dict,
            incoming_random_keys_dict,
            outgoing_random_keys_dict,
            param_dict,
        ) = self.create_masks(param_dict, activations_score_dict, key)

        params = unflatten_dict(param_dict, sep="/")
        incoming_random_keys = unflatten_dict(incoming_random_keys_dict, sep="/")
        if self.init_method_outgoing == "random":
            outgoing_random_keys = unflatten_dict(outgoing_random_keys_dict, sep="/")

        incoming_mask = unflatten_dict(incoming_mask_dict, sep="/")
        reinit_fn = functools.partial(
            weight_reinit_random,
            weight_scaling=self.weight_scaling,
            scale=self.incoming_scale,
            weights_type="incoming",
        )
        weight_random_reset_fn = jax.jit(functools.partial(jax.tree.map, reinit_fn))
        params = weight_random_reset_fn(params, incoming_mask, incoming_random_keys)

        outgoing_mask = unflatten_dict(outgoing_mask_dict, sep="/")
        if self.init_method_outgoing == "random":
            reinit_fn = functools.partial(
                weight_reinit_random,
                weight_scaling=self.weight_scaling,
                scale=self.outgoing_scale,
                weights_type="outgoing",
            )
            weight_random_reset_fn = jax.jit(functools.partial(jax.tree.map, reinit_fn))
            params = weight_random_reset_fn(params, outgoing_mask, outgoing_random_keys)
        elif self.init_method_outgoing == "zero":
            weight_zero_reset_fn = jax.jit(
                functools.partial(jax.tree.map, weight_reinit_zero)
            )
            params = weight_zero_reset_fn(params, outgoing_mask)
        else:
            raise ValueError(f"Invalid init method: {self.init_method_outgoing}")

        reset_momentum_fn = jax.jit(functools.partial(jax.tree.map, reset_momentum))

        # Modify Adam state updating to work with optax Adam trace and momentum
        opt_state_list = list(opt_state)
        # Find which element of opt_state is ScaleByAdamState
        adam_state_idx = -1
        for i, s in enumerate(opt_state_list):
            if isinstance(s, optax.ScaleByAdamState):
                adam_state_idx = i
                break

        if adam_state_idx != -1:
            adam_state = opt_state_list[adam_state_idx]
            new_mu = reset_momentum_fn(adam_state.mu, incoming_mask)
            new_mu = reset_momentum_fn(new_mu, outgoing_mask)
            new_nu = reset_momentum_fn(adam_state.nu, incoming_mask)
            new_nu = reset_momentum_fn(new_nu, outgoing_mask)

            opt_state_list[adam_state_idx] = optax.ScaleByAdamState(
                adam_state.count, mu=new_mu, nu=new_nu
            )

        opt_state = tuple(opt_state_list)
        return params, opt_state

    def _score2mask(self, activation, param, next_param, key):
        del key, param, next_param
        score = self.estimate_neuron_score(activation)
        return score <= self.dead_neurons_threshold

    def create_masks(self, param_dict, activations_dict, key):
        incoming_mask_dict = {
            k: jnp.zeros_like(p) if p.ndim != 1 else None for k, p in param_dict.items()
        }
        outgoing_mask_dict = {
            k: jnp.zeros_like(p) if p.ndim != 1 else None for k, p in param_dict.items()
        }
        ingoing_random_keys_dict = {k: None for k in param_dict}
        outgoing_random_keys_dict = (
            {k: None for k in param_dict}
            if self.init_method_outgoing == "random"
            else {}
        )

        for k in self.reset_layers:
            # param_dict structure for Haiku: phi/phi/w
            param_key = f"phi/{k}/w"  # In representation networks, parameters are generally under phi group
            if param_key not in param_dict:  # fallback if structured differently
                param_key = f"{k}/w"
                if param_key not in param_dict:
                    continue

            param = param_dict[param_key]

            next_key = self.next_layers[k]
            if isinstance(next_key, list):
                next_key = next_key[0]

            next_param_key = f"phi/{next_key}/w"
            if next_param_key not in param_dict:
                next_param_key = f"{next_key}/w"
                if next_param_key not in param_dict:
                    # In DQN, last layer might cross to 'q' head. We map it by default
                    if next_key == "q":
                        next_param_key = f"q/q/w"
                    else:
                        continue

            next_param = param_dict[next_param_key]

            activation = activations_dict[k]
            # Haiku intercept doesn't return batch arrays inside tuples, just the jax array
            # and they are pre-ReLU usually in our code. So we apply ReLU here before computing deadness
            activation = jax.nn.relu(activation)

            neuron_mask = self._score2mask(activation, param, next_param, key)
            is_dead = neuron_mask

            if is_dead.sum() == 0:
                continue

            # Current parameter processing (incoming weights)
            if param.ndim == 2:  # Linear layer
                incoming_mask_dict[param_key] = (
                    incoming_mask_dict[param_key]
                    .at[:, is_dead]  # Zero out axis 1 (output dim) for current layer
                    .set(1.0)
                )
            elif param.ndim == 4:  # Conv layer
                incoming_mask_dict[param_key] = (
                    incoming_mask_dict[param_key]
                    .at[:, :, :, is_dead]  # Zero out output channels
                    .set(1.0)
                )

            # Next parameter processing (outgoing weights)
            next_keys = (
                self.next_layers[k]
                if isinstance(self.next_layers[k], list)
                else [self.next_layers[k]]
            )
            for next_k in next_keys:
                np_key = f"phi/{next_k}/w"
                if np_key not in param_dict:
                    np_key = f"{next_k}/w"
                    if next_k == "q":
                        np_key = f"q/q/w"

                next_param = param_dict[np_key]
                if next_param.ndim == 2:  # Linear layer
                    if param.ndim == 4:  # Conv followed by Linear (after Flatten)
                        C = param.shape[-1]
                        H_W_C = next_param.shape[0]
                        H_W = H_W_C // C
                        expanded_is_dead = jnp.tile(is_dead, H_W)
                        outgoing_mask_dict[np_key] = (
                            outgoing_mask_dict[np_key].at[expanded_is_dead, :].set(1.0)
                        )
                    else:
                        outgoing_mask_dict[np_key] = (
                            outgoing_mask_dict[np_key]
                            .at[
                                is_dead, :
                            ]  # Zero out axis 0 (input dim) for next layer
                            .set(1.0)
                        )
                elif next_param.ndim == 4:  # Conv layer
                    outgoing_mask_dict[np_key] = (
                        outgoing_mask_dict[np_key]
                        .at[is_dead, :, :, :]  # Zero out input channels
                        .set(1.0)
                    )

            key, subkey = random.split(key)
            ingoing_random_keys_dict[param_key] = subkey
            if self.init_method_outgoing == "random":
                key, subkey = random.split(key)
                outgoing_random_keys_dict[np_key] = subkey

            bias_key = param_key.replace("/w", "/b")
            if bias_key in param_dict:
                new_bias = jnp.zeros_like(param_dict[bias_key])
                param_dict[bias_key] = jnp.where(
                    neuron_mask, new_bias, param_dict[bias_key]
                )

        return (
            incoming_mask_dict,
            outgoing_mask_dict,
            ingoing_random_keys_dict,
            outgoing_random_keys_dict,
            param_dict,
        )

    def create_mask_helper(self, neuron_mask, current_param, next_param):
        def mask_creator(expansion_axis, expansion_axes, param, neuron_mask):
            if param.ndim == 2:
                axes = expansion_axis
                if axes[0] == 1 and (param.shape[0] > neuron_mask.shape[0]):
                    num_repeatition = int(param.shape[0] / neuron_mask.shape[0])
                    neuron_mask = jnp.repeat(neuron_mask, num_repeatition, axis=0)
            elif param.ndim == 4:
                axes = expansion_axes
            mask = jnp.expand_dims(neuron_mask, axis=tuple(axes))
            for i in range(len(axes)):
                mask = jnp.repeat(mask, param.shape[axes[i]], axis=axes[i])
            return mask

        incoming_mask = mask_creator([0], [0, 1, 2], current_param, neuron_mask)
        outgoing_mask = mask_creator([1], [0, 1, 3], next_param, neuron_mask)
        return incoming_mask, outgoing_mask


class NeuronRecyclerScheduled(NeuronRecycler):
    def __init__(
        self,
        *args,
        score_type="redo",
        recycle_rate=0.3,
        **kwargs,
    ):
        super(NeuronRecyclerScheduled, self).__init__(*args, **kwargs)
        self.score_type = score_type
        self.recycle_rate = recycle_rate

    def _score2mask(self, activation, param, next_param, key):
        is_cbp = self.score_type == "cbp"
        score = self.estimate_neuron_score(activation, is_cbp=is_cbp)
        if self.score_type == "redo":
            pass
        elif self.score_type == "random":
            new_key = random.fold_in(key, self._last_update_step)
            score = random.permutation(new_key, score, independent=True)
        elif self.score_type == "redo_inverted":
            score = -score
        elif self.score_type == "cbp":
            next_axes = list(range(param.ndim))
            del next_axes[-2]
            current_axes = list(range(param.ndim))
            del current_axes[-1]
            if next_param.ndim == 2 and param.ndim == 4:
                new_shape = activation.shape[1:] + (-1,)
                next_param = jnp.reshape(next_param, new_shape)
            score *= jnp.sum(jnp.abs(next_param), axis=next_axes) / (
                jnp.sum(jnp.abs(param), axis=current_axes) + 1e-9
            )
        # ReDo logic: recycle neurons with score < threshold, up to p*N neurons (p = recycle_rate)
        candidate_mask = leastk_mask(score, self.recycle_rate)
        is_below_threshold = score <= self.dead_neurons_threshold
        return (candidate_mask > 0) & is_below_threshold
