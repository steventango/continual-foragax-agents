from dataclasses import replace
from functools import partial
from typing import Dict

import jax
import jax.lax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN import AgentState as BaseAgentState
from algorithms.nn.DQN import Hypers as BaseHypers
from algorithms.nn.NNAgent import Metrics
from utils.weight_recyclers_hk import NeuronRecyclerScheduled


@cxu.dataclass
class Hypers(BaseHypers):
    reset_period: int
    cycle_steps: int
    recycle_rate: float
    score_type: str


@cxu.dataclass
class AgentState(BaseAgentState):
    hypers: Hypers


class DQN_ReDo(DQN):
    def __init__(
        self,
        observations: tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)

        hypers = Hypers(
            **self.state.hypers.__dict__,
            reset_period=params.get(
                "reset_period", 200_000
            ),  # Dopamine defaults to 200k
            cycle_steps=params.get("cycle_steps", 10_000_000),
            recycle_rate=params.get("recycle_rate", 0.3),
            score_type=params.get("score_type", "redo"),
        )

        # ReDo typically resets layers in the feature network. We will get names from the params.
        all_layers = []
        for k in self.state.params["phi"].keys():
            all_layers.append(k)

        all_layers.append(
            "q"
        )  # Append q head as the output layer so last layer in phi knows outgoing

        self.recycler = NeuronRecyclerScheduled(
            all_layers_names=all_layers,
            reset_period=hypers.reset_period,
            reset_start_step=0,
            reset_end_step=hypers.cycle_steps,
            score_type=hypers.score_type,
            recycle_rate=hypers.recycle_rate,
            logging_period=hypers.reset_period,  # Let's log dead neurons alongside resets
            dead_neurons_threshold=0.025,
        )

        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            hypers=hypers,
        )

    @partial(jax.jit, static_argnums=0)
    def _maybe_update(self, state: AgentState) -> AgentState:
        # Standard _maybe_update will increment state.steps and update if needed
        # We will also intercept the update to get intermediates, or just run a forward pass
        # when a reset is needed to get the activations over a batch.

        def do_update_and_reset():
            new_state, metrics = self._update_with_redo(state)
            metrics = {
                "weight_change": metrics.get(
                    "weight_change", state.metrics.weight_change
                ),
                "abs_td_error": metrics.get("abs_td_error", state.metrics.abs_td_error),
                "squared_td_error": metrics.get(
                    "squared_td_error", state.metrics.squared_td_error
                ),
                "loss": metrics.get("loss", state.metrics.loss),
            }
            # Add redo specific metrics
            for k, v in metrics.items():
                if k.startswith("dead_"):
                    metrics[k] = v

            metrics_obj = Metrics(
                weight_change=metrics["weight_change"],
                abs_td_error=metrics["abs_td_error"],
                squared_td_error=metrics["squared_td_error"],
                loss=metrics["loss"],
                dead_feature_percentage=metrics.get(
                    "dead_feature_percentage", state.metrics.dead_feature_percentage
                ),
            )
            new_state = replace(new_state, metrics=metrics_obj)
            return new_state, metrics

        def do_update():
            new_state, metrics = self._update(state)

            metrics_obj = Metrics(
                weight_change=metrics.get("weight_change", state.metrics.weight_change),
                abs_td_error=metrics.get("abs_td_error", state.metrics.abs_td_error),
                squared_td_error=metrics.get(
                    "squared_td_error", state.metrics.squared_td_error
                ),
                loss=metrics.get("loss", state.metrics.loss),
                dead_feature_percentage=metrics.get(
                    "dead_feature_percentage", state.metrics.dead_feature_percentage
                ),
            )
            new_state = replace(new_state, metrics=metrics_obj)
            return new_state, metrics

        def no_update():
            return state, {}

        # Is it a reset step?
        is_reset_step = (state.steps > 0) & (
            state.steps % state.hypers.reset_period == 0
        )

        new_state, step_metrics = jax.lax.cond(
            (state.steps % state.hypers.update_freq == 0)
            & self.buffer.can_sample(state.buffer_state),
            lambda: jax.lax.cond(is_reset_step, do_update_and_reset, do_update),
            no_update,
        )

        new_state = replace(new_state, steps=state.steps + 1)
        new_state = self._decay_epsilon(new_state)

        jax.debug.callback(self._log_metrics_if_exist, step_metrics)
        return new_state

    def _log_metrics_if_exist(self, metrics):
        if metrics and "dead_feature_percentage" in metrics:
            for k, v in metrics.items():
                if k.startswith("dead_"):
                    self.collector.collect(k, v)

    @partial(jax.jit, static_argnums=0)
    def _update_with_redo(self, state: AgentState):
        updates = state.updates + 1
        state.key, buffer_sample_key = jax.random.split(state.key)
        batch = self.buffer.sample(state.buffer_state, buffer_sample_key)

        # 1. Forward pass to get intermediates
        x = batch.experience["x"][:, 0]
        phi_out = self.builder._feat_net.apply(state.params["phi"], x)

        intermediates = phi_out.activations

        # 2. Recycle weights
        state.key, reset_key = jax.random.split(state.key)

        new_params, new_opt_state = self.recycler.maybe_update_weights(
            state.steps, intermediates, state.params, reset_key, state.optim
        )

        # 3. Compute metric for dead neurons
        dead_logs = self.recycler.maybe_log_deadneurons(state.steps, intermediates)

        state = replace(state, params=new_params, optim=new_opt_state)

        # 4. Standard update
        state, metrics = self._computeUpdate(state, batch.experience)

        if dead_logs is not None:
            metrics.update(dead_logs)

        target_params = self._update_target_network(state, updates)

        return replace(
            state,
            updates=updates,
            target_params=target_params,
        ), metrics
