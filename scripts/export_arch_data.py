"""
Export architecture data for all variants at FOV=9 to public/arch_data.json.
Includes per-layer param counts, graph nodes/edges, and pre-generated Mermaid source.

Run with: uv run scripts/export_arch_data.py
"""

import os
import sys
import json
import jax
import jax.numpy as jnp
import haiku as hk

sys.path.append(os.path.abspath("src"))

from algorithms.PPORegistry import getAgent as getPPOAgent
from experiment.ExperimentModel import load as load_exp
from representations.networks import NetworkBuilder

FOV = 9
OBS_SHAPE = (FOV, FOV, 3)
ACTIONS = 4
BASE = f"experiments/E136-big/foragax-sweep/ForagaxBig-v4/{FOV}"

CLASSDEFS = """                        classDef styleInput fill:#fbcfe8,stroke:#f472b6,color:#0f172a
                        classDef styleNorm fill:#fef9c3,stroke:#facc15,color:#0f172a
                        classDef styleMlp fill:#e0f2fe,stroke:#38bdf8,color:#0f172a
                        classDef styleRnn fill:#ffedd5,stroke:#fb923c,color:#0f172a
                        classDef styleHead fill:#fae8ff,stroke:#e879f9,color:#0f172a
                        classDef styleConcat fill:#f8fafc,stroke:#64748b,stroke-dasharray: 4 4,color:#0f172a"""


def p(n):
    """Format param count."""
    return f"{n:,}"


def count_params(params):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def params_by_module(params_dict):
    """Flatten nested Flax params dict to {module_name: count}."""
    result = {}

    def _walk(d, prefix=()):
        for k, v in d.items():
            path = prefix + (k,)
            if isinstance(v, dict):
                _walk(v, path)
            else:
                mod = "/".join(prefix) if prefix else k
                result[mod] = result.get(mod, 0) + int(v.size)

    top = params_dict.get("params", params_dict)
    _walk(top)
    return result


def graph_to_mermaid(graph):
    """Convert a graph dict to a Mermaid flowchart string."""
    in_sg = {}
    for sg in graph.get("subgraphs", []):
        for nid in sg["nodes"]:
            in_sg[nid] = sg["id"]

    lines = ["graph TD", CLASSDEFS]

    # Top-level nodes (not in any subgraph)
    for nid, node in graph["nodes"].items():
        if nid not in in_sg:
            style = node.get("type", "styleMlp")
            lines.append(f'                        {nid}("{node["label"]}"):::{style}')

    # Subgraphs
    for sg in graph.get("subgraphs", []):
        lines.append(f'                        subgraph {sg["id"]} ["{sg["label"]}"]')
        lines.append("                        direction TB")
        for nid in sg["nodes"]:
            node = graph["nodes"][nid]
            style = node.get("type", "styleMlp")
            lines.append(f'                        {nid}("{node["label"]}"):::{style}')
        # Intra-subgraph edges
        for a, b in graph["edges"]:
            if in_sg.get(a) == sg["id"] and in_sg.get(b) == sg["id"]:
                lines.append(f"                        {a} --> {b}")
        lines.append("                        end")

    # Cross-subgraph and external edges
    for a, b in graph["edges"]:
        if in_sg.get(a) != in_sg.get(b) or not in_sg.get(a):
            lines.append(f"                        {a} --> {b}")

    # Invisible ordering edges
    for a, b in graph.get("invisible_edges", []):
        lines.append(f"                        {a} ~~~ {b}")

    return "\n".join(lines)


def _lbl(name, dims=None, params=None):
    """Build a node label: Name / dims / params.
    Avoids parentheses in output since they confuse the Mermaid parser.
    """
    parts = [name]
    if dims:
        # Replace () with [] to avoid Mermaid parse errors
        parts.append(dims.replace("(", "[").replace(")", "]"))
    if params:
        parts.append(f"{p(params)} params")
    return "\\n".join(parts)


# ── DQN graph builders ────────────────────────────────────────────────────────


def build_dqn_graph(variant_data):
    m = variant_data["modules"]
    use_ln = variant_data["use_layernorm"]
    has_rt = variant_data["has_rt"]
    conv_out = variant_data["conv_out"]  # 1296
    concat_sz = variant_data["concat_size"]  # 1301 or 1302
    scalars = variant_data["scalars"]

    scalar_label = "Last A, Last R, R Trace" if has_rt else "Last Action + Last Reward"
    scalar_dims = f"({scalars})"

    conv_p = m.get("phi/phi/~/phi", 0)
    d1_p = m.get("phi/phi/~/linear", 0)
    d2_p = m.get("phi/phi/~/linear_1", 0)
    qh_p = m.get("q/q", 0)

    nodes = {
        "Obs": {
            "label": _lbl(f"Observation", f"{FOV}×{FOV}×3"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Scalars": {
            "label": _lbl(scalar_label, scalar_dims),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Conv": {
            "label": _lbl("Conv2D 16×3×3", f"{FOV}×{FOV}×3→{FOV}×{FOV}×16", conv_p),
            "type": "styleMlp",
        },
        "ReLU1": {"label": "ReLU", "type": "styleNorm"},
        "Flat": {"label": _lbl("Flatten", f"→{conv_out}"), "type": "styleMlp"},
        "Concat": {"label": _lbl("Concat", f"→{concat_sz}"), "type": "styleConcat"},
        "Dense1": {"label": _lbl("Dense", f"{concat_sz}→64", d1_p), "type": "styleMlp"},
        "Dense2": {"label": _lbl("Dense", "64→64", d2_p), "type": "styleMlp"},
        "QHead": {"label": _lbl("Q-Head", "64→4", qh_p), "type": "styleHead"},
    }
    edges = [
        ("Obs", "Conv"),
        ("Conv", "ReLU1"),
        ("ReLU1", "Flat"),
        ("Flat", "Concat"),
        ("Scalars", "Concat"),
        ("Concat", "Dense1"),
    ]
    repr_nodes = ["Dense1"]

    if use_ln:
        ln1_p = m.get("phi/phi/~/layer_norm", 0)
        ln2_p = m.get("phi/phi/~/layer_norm_1", 0)
        nodes["LN1"] = {"label": _lbl("LayerNorm", "64", ln1_p), "type": "styleNorm"}
        nodes["Act1"] = {"label": "ReLU", "type": "styleNorm"}
        nodes["LN2"] = {"label": _lbl("LayerNorm", "64", ln2_p), "type": "styleNorm"}
        nodes["Act2"] = {"label": "ReLU", "type": "styleNorm"}
        edges += [
            ("Dense1", "LN1"),
            ("LN1", "Act1"),
            ("Act1", "Dense2"),
            ("Dense2", "LN2"),
            ("LN2", "Act2"),
            ("Act2", "QHead"),
        ]
        repr_nodes += ["LN1", "Act1", "Dense2", "LN2", "Act2"]
    else:
        nodes["Act1"] = {"label": "ReLU", "type": "styleNorm"}
        nodes["Act2"] = {"label": "ReLU", "type": "styleNorm"}
        edges += [
            ("Dense1", "Act1"),
            ("Act1", "Dense2"),
            ("Dense2", "Act2"),
            ("Act2", "QHead"),
        ]
        repr_nodes += ["Act1", "Dense2", "Act2"]

    # Reorder nodes so subgraph content appears contiguous
    ordered_nodes = {}
    top_keys = ["Obs", "Scalars", "Conv", "ReLU1", "Flat", "Concat"]
    for k in top_keys:
        ordered_nodes[k] = nodes[k]
    for k in repr_nodes:
        ordered_nodes[k] = nodes[k]
    ordered_nodes["QHead"] = nodes["QHead"]

    return {
        "nodes": ordered_nodes,
        "edges": edges,
        "subgraphs": [
            {
                "id": "ForagerNet",
                "label": "ForagerNet (Representation)",
                "nodes": repr_nodes,
            }
        ],
    }


# ── DRQN graph builders ───────────────────────────────────────────────────────


def build_drqn_graph(variant_data):
    m = variant_data["modules"]
    use_ln = variant_data["use_layernorm"]
    conv_out = variant_data["conv_out"]
    concat_sz = variant_data["concat_size"]
    pre = variant_data["pre_gru_layers"]

    conv_p = m.get("phi/ForagerGRUNetReLU/~/phi", 0)
    gru_p = m.get("phi/ForagerGRUNetReLU/~/gru", 0) + m.get(
        "phi/ForagerGRUNetReLU/~/gru/~/gru_inner", 0
    )
    qh_p = m.get("q/q", 0)

    # GRU input and output dims
    if pre > 0:
        pre_p = m.get("phi/ForagerGRUNetReLU/~/linear", 0)
        gru_in = 64
        skip_in = 64  # pre-GRU output feeds into skip
        skip_sz = gru_in + 64  # 128
        # Q-head input = skip_sz = 128
    else:
        pre_p = 0
        gru_in = concat_sz
        skip_in = concat_sz
        skip_sz = gru_in + concat_sz  # 64 + 1301 = 1365
        # Q-head input = skip_sz

    nodes = {
        "Obs": {
            "label": _lbl("Seq Obs", f"{FOV}×{FOV}×3"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Scalars": {
            "label": _lbl("Seq Scalars", f"({variant_data['scalars']})"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Conv": {
            "label": _lbl("Conv2D 16×3×3", f"{FOV}×{FOV}×3→{FOV}×{FOV}×16", conv_p),
            "type": "styleMlp",
        },
        "ReLU1": {"label": "ReLU", "type": "styleNorm"},
        "Flat": {"label": _lbl("Flatten", f"→{conv_out}"), "type": "styleMlp"},
        "Concat": {"label": _lbl("Concat", f"→{concat_sz}"), "type": "styleConcat"},
        "GRU": {"label": _lbl(f"GRU 64", f"{gru_in}→64", gru_p), "type": "styleRnn"},
        "ReLU2": {"label": "ReLU", "type": "styleNorm"},
        "Skip": {"label": _lbl("Concat Skip", f"→{skip_sz}"), "type": "styleConcat"},
        "QHead": {"label": _lbl("Q-Head", f"{skip_sz}→4", qh_p), "type": "styleHead"},
    }
    edges = [
        ("Obs", "Conv"),
        ("Conv", "ReLU1"),
        ("ReLU1", "Flat"),
        ("Flat", "Concat"),
        ("Scalars", "Concat"),
    ]

    pre_nodes = []
    if pre > 0:
        ln_p = m.get("phi/ForagerGRUNetReLU/~/layer_norm", 0) if use_ln else 0
        nodes["Pre"] = {
            "label": _lbl("Dense", f"{concat_sz}→64", pre_p),
            "type": "styleMlp",
        }
        pre_nodes.append("Pre")
        if use_ln:
            nodes["PreLN"] = {
                "label": _lbl("LayerNorm", "64", ln_p),
                "type": "styleNorm",
            }
            nodes["PreAct"] = {"label": "ReLU", "type": "styleNorm"}
            edges += [
                ("Concat", "Pre"),
                ("Pre", "PreLN"),
                ("PreLN", "PreAct"),
                ("PreAct", "GRU"),
            ]
            edges.append(("PreAct", "Skip"))
            pre_nodes += ["PreLN", "PreAct"]
        else:
            nodes["PreAct"] = {"label": "ReLU", "type": "styleNorm"}
            edges += [("Concat", "Pre"), ("Pre", "PreAct"), ("PreAct", "GRU")]
            edges.append(("PreAct", "Skip"))
            pre_nodes.append("PreAct")
    else:
        edges.append(("Concat", "GRU"))
        edges.append(("Concat", "Skip"))  # skip gets the raw concat

    edges += [("GRU", "ReLU2"), ("ReLU2", "Skip"), ("Skip", "QHead")]

    # Post-GRU nodes (if any, none in current configs after review)

    subgraphs = []
    if pre_nodes:
        subgraphs.append({"id": "PreGRU", "label": "Pre-GRU", "nodes": pre_nodes})

    return {
        "nodes": nodes,
        "edges": edges,
        "subgraphs": subgraphs,
    }


# ── PPO graph builders ────────────────────────────────────────────────────────


def build_ppo_graph(variant_data):
    m = variant_data["modules"]
    use_ln = variant_data["use_layernorm"]
    use_rt = variant_data["use_reward_trace"]
    obs_flat = variant_data["obs_flat"]  # 243
    obs_hid = variant_data["obs_hidden"]  # 59
    cat_sz = variant_data["concat_size"]  # 64
    d_hid = variant_data["d_hidden"]  # 192
    hid = variant_data["hidden_size"]  # 64
    scalars_count = cat_sz - obs_hid  # 5 or 6

    scalar_label = "Last A, Last R, R Trace" if use_rt else "Last Action + Last Reward"

    ad1 = m.get("actor_dense1", 0)
    cd1 = m.get("critic_dense1", 0)
    ad2 = m.get("actor_dense2", 0)
    cd2 = m.get("critic_dense2", 0)
    ad3 = m.get("actor_dense3", 0)
    cd3 = m.get("critic_dense3", 0)
    aln1 = m.get("actor_layernorm1", 0)
    cln1 = m.get("critic_layernorm1", 0)
    aln2 = m.get("actor_layernorm2", 0)
    cln2 = m.get("critic_layernorm2", 0)
    ahead = m.get("actor_mean", 0)
    chead = m.get("critic_value", 0)

    nodes = {
        "Obs": {
            "label": _lbl("Observation", f"{FOV}×{FOV}×3"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Scalars": {
            "label": _lbl(scalar_label, f"({scalars_count})"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Flat": {"label": _lbl("Flatten", f"→{obs_flat}"), "type": "styleMlp"},
        # Actor path
        "A_Emb": {
            "label": _lbl("Dense", f"{obs_flat}→{obs_hid}", ad1),
            "type": "styleMlp",
        },
        "A_Concat": {"label": _lbl("Concat", f"→{cat_sz}"), "type": "styleConcat"},
        "A_Dense2": {
            "label": _lbl("Dense", f"{cat_sz}→{d_hid}", ad2),
            "type": "styleMlp",
        },
        "A_Dense3": {"label": _lbl("Dense", f"{d_hid}→{hid}", ad3), "type": "styleMlp"},
        "A_Act3": {"label": "Tanh", "type": "styleNorm"},
        "A_Head": {"label": _lbl("Logits", f"{hid}→4", ahead), "type": "styleHead"},
        # Critic path
        "C_Emb": {
            "label": _lbl("Dense", f"{obs_flat}→{obs_hid}", cd1),
            "type": "styleMlp",
        },
        "C_Concat": {"label": _lbl("Concat", f"→{cat_sz}"), "type": "styleConcat"},
        "C_Dense2": {
            "label": _lbl("Dense", f"{cat_sz}→{d_hid}", cd2),
            "type": "styleMlp",
        },
        "C_Dense3": {"label": _lbl("Dense", f"{d_hid}→{hid}", cd3), "type": "styleMlp"},
        "C_Act3": {"label": "Tanh", "type": "styleNorm"},
        "C_Head": {"label": _lbl("Value", f"{hid}→1", chead), "type": "styleHead"},
    }

    actor_nodes = ["A_Emb", "A_Concat", "A_Dense2", "A_Dense3", "A_Act3", "A_Head"]
    critic_nodes = ["C_Emb", "C_Concat", "C_Dense2", "C_Dense3", "C_Act3", "C_Head"]
    edges = [
        ("Obs", "Flat"),
        ("Flat", "A_Emb"),
        ("Flat", "C_Emb"),
        ("Scalars", "A_Concat"),
        ("Scalars", "C_Concat"),
    ]

    if use_ln:
        nodes["A_LN1"] = {
            "label": _lbl("LayerNorm", f"{obs_hid}", aln1),
            "type": "styleNorm",
        }
        nodes["A_Act1"] = {"label": "Tanh", "type": "styleNorm"}
        nodes["A_LN2"] = {
            "label": _lbl("LayerNorm", f"{hid}", aln2),
            "type": "styleNorm",
        }
        nodes["C_LN1"] = {
            "label": _lbl("LayerNorm", f"{obs_hid}", cln1),
            "type": "styleNorm",
        }
        nodes["C_Act1"] = {"label": "Tanh", "type": "styleNorm"}
        nodes["C_LN2"] = {
            "label": _lbl("LayerNorm", f"{hid}", cln2),
            "type": "styleNorm",
        }
        edges += [
            ("A_Emb", "A_LN1"),
            ("A_LN1", "A_Act1"),
            ("A_Act1", "A_Concat"),
            ("A_Concat", "A_Dense2"),
            ("A_Dense2", "A_Dense3"),
            ("A_Dense3", "A_LN2"),
            ("A_LN2", "A_Act3"),
            ("A_Act3", "A_Head"),
            ("C_Emb", "C_LN1"),
            ("C_LN1", "C_Act1"),
            ("C_Act1", "C_Concat"),
            ("C_Concat", "C_Dense2"),
            ("C_Dense2", "C_Dense3"),
            ("C_Dense3", "C_LN2"),
            ("C_LN2", "C_Act3"),
            ("C_Act3", "C_Head"),
        ]
        actor_nodes = [
            "A_Emb",
            "A_LN1",
            "A_Act1",
            "A_Concat",
            "A_Dense2",
            "A_Dense3",
            "A_LN2",
            "A_Act3",
            "A_Head",
        ]
        critic_nodes = [
            "C_Emb",
            "C_LN1",
            "C_Act1",
            "C_Concat",
            "C_Dense2",
            "C_Dense3",
            "C_LN2",
            "C_Act3",
            "C_Head",
        ]
    else:
        nodes["A_Act1"] = {"label": "Tanh", "type": "styleNorm"}
        nodes["C_Act1"] = {"label": "Tanh", "type": "styleNorm"}
        edges += [
            ("A_Emb", "A_Act1"),
            ("A_Act1", "A_Concat"),
            ("A_Concat", "A_Dense2"),
            ("A_Dense2", "A_Dense3"),
            ("A_Dense3", "A_Act3"),
            ("A_Act3", "A_Head"),
            ("C_Emb", "C_Act1"),
            ("C_Act1", "C_Concat"),
            ("C_Concat", "C_Dense2"),
            ("C_Dense2", "C_Dense3"),
            ("C_Dense3", "C_Act3"),
            ("C_Act3", "C_Head"),
        ]
        actor_nodes = [
            "A_Emb",
            "A_Act1",
            "A_Concat",
            "A_Dense2",
            "A_Dense3",
            "A_Act3",
            "A_Head",
        ]
        critic_nodes = [
            "C_Emb",
            "C_Act1",
            "C_Concat",
            "C_Dense2",
            "C_Dense3",
            "C_Act3",
            "C_Head",
        ]

    return {
        "nodes": nodes,
        "edges": edges,
        "subgraphs": [
            {"id": "Actor", "label": "Actor", "nodes": actor_nodes},
            {"id": "Critic", "label": "Critic", "nodes": critic_nodes},
        ],
        "invisible_edges": [["Actor", "Critic"]],
    }


def build_ppo_rtu_graph(variant_data):
    m = variant_data["modules"]
    use_ln = variant_data["use_layernorm"]
    use_rt = variant_data["use_reward_trace"]
    obs_flat = variant_data["obs_flat"]  # 243
    obs_hid = variant_data["obs_hidden"]  # 59
    cat_sz = variant_data["concat_size"]  # 64
    d_hid = variant_data["d_hidden"]  # 192 or 512
    hid = variant_data["hidden_size"]  # 64
    scalars_count = cat_sz - obs_hid

    scalar_label = "Last A, Last R, R Trace" if use_rt else "Last Action + Last Reward"

    ad1 = m.get("actor_dense1", 0)
    cd1 = m.get("critic_dense1", 0)
    # RTU total params
    artu = sum(v for k, v in m.items() if k.startswith("actor_rtu"))
    crtu = sum(v for k, v in m.items() if k.startswith("critic_rtu"))
    ad2 = m.get("actor_dense2", 0)
    cd2 = m.get("critic_dense2", 0)
    aln1 = m.get("actor_layernorm1", 0)
    cln1 = m.get("critic_layernorm1", 0)
    aln2 = m.get("actor_layernorm2", 0)
    cln2 = m.get("critic_layernorm2", 0)
    ahead = m.get("actor_mean", 0)
    chead = m.get("critic_value", 0)

    # RTU outputs 2×d_hidden, then concat with skip(cat_sz) → 2*d_hid + cat_sz
    rtu_out = 2 * d_hid
    skip_out = rtu_out + cat_sz

    nodes = {
        "Obs": {
            "label": _lbl("Observation", f"{FOV}×{FOV}×3"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Scalars": {
            "label": _lbl(scalar_label, f"({scalars_count})"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Flat": {"label": _lbl("Flatten", f"→{obs_flat}"), "type": "styleMlp"},
        # Actor
        "A_Emb": {
            "label": _lbl("Dense", f"{obs_flat}→{obs_hid}", ad1),
            "type": "styleMlp",
        },
        "A_Concat": {"label": _lbl("Concat", f"→{cat_sz}"), "type": "styleConcat"},
        "A_RTU": {
            "label": _lbl(f"RTU {d_hid}×2", f"{cat_sz}→{rtu_out}", artu),
            "type": "styleRnn",
        },
        "A_Skip": {"label": _lbl("Join Skip", f"→{skip_out}"), "type": "styleConcat"},
        "A_Dense2": {
            "label": _lbl("Dense", f"{skip_out}→{hid}", ad2),
            "type": "styleMlp",
        },
        "A_Act2": {"label": "Tanh", "type": "styleNorm"},
        "A_Head": {"label": _lbl("Logits", f"{hid}→4", ahead), "type": "styleHead"},
        # Critic
        "C_Emb": {
            "label": _lbl("Dense", f"{obs_flat}→{obs_hid}", cd1),
            "type": "styleMlp",
        },
        "C_Concat": {"label": _lbl("Concat", f"→{cat_sz}"), "type": "styleConcat"},
        "C_RTU": {
            "label": _lbl(f"RTU {d_hid}×2", f"{cat_sz}→{rtu_out}", crtu),
            "type": "styleRnn",
        },
        "C_Skip": {"label": _lbl("Join Skip", f"→{skip_out}"), "type": "styleConcat"},
        "C_Dense2": {
            "label": _lbl("Dense", f"{skip_out}→{hid}", cd2),
            "type": "styleMlp",
        },
        "C_Act2": {"label": "Tanh", "type": "styleNorm"},
        "C_Head": {"label": _lbl("Value", f"{hid}→1", chead), "type": "styleHead"},
    }
    actor_nodes = [
        "A_Emb",
        "A_Concat",
        "A_RTU",
        "A_Skip",
        "A_Dense2",
        "A_Act2",
        "A_Head",
    ]
    critic_nodes = [
        "C_Emb",
        "C_Concat",
        "C_RTU",
        "C_Skip",
        "C_Dense2",
        "C_Act2",
        "C_Head",
    ]

    edges = [
        ("Obs", "Flat"),
        ("Flat", "A_Emb"),
        ("Flat", "C_Emb"),
        ("Scalars", "A_Concat"),
        ("Scalars", "C_Concat"),
        ("A_Concat", "A_RTU"),
        ("A_Concat", "A_Skip"),
        ("A_RTU", "A_Skip"),
        ("A_Skip", "A_Dense2"),
        ("C_Concat", "C_RTU"),
        ("C_Concat", "C_Skip"),
        ("C_RTU", "C_Skip"),
        ("C_Skip", "C_Dense2"),
    ]

    if use_ln:
        nodes["A_LN1"] = {
            "label": _lbl("LayerNorm", f"{obs_hid}", aln1),
            "type": "styleNorm",
        }
        nodes["A_Act1"] = {"label": "Tanh", "type": "styleNorm"}
        nodes["A_LN2"] = {
            "label": _lbl("LayerNorm", f"{hid}", aln2),
            "type": "styleNorm",
        }
        nodes["C_LN1"] = {
            "label": _lbl("LayerNorm", f"{obs_hid}", cln1),
            "type": "styleNorm",
        }
        nodes["C_Act1"] = {"label": "Tanh", "type": "styleNorm"}
        nodes["C_LN2"] = {
            "label": _lbl("LayerNorm", f"{hid}", cln2),
            "type": "styleNorm",
        }
        edges += [
            ("A_Emb", "A_LN1"),
            ("A_LN1", "A_Act1"),
            ("A_Act1", "A_Concat"),
            ("A_Dense2", "A_LN2"),
            ("A_LN2", "A_Act2"),
            ("A_Act2", "A_Head"),
            ("C_Emb", "C_LN1"),
            ("C_LN1", "C_Act1"),
            ("C_Act1", "C_Concat"),
            ("C_Dense2", "C_LN2"),
            ("C_LN2", "C_Act2"),
            ("C_Act2", "C_Head"),
        ]
        actor_nodes = [
            "A_Emb",
            "A_LN1",
            "A_Act1",
            "A_Concat",
            "A_RTU",
            "A_Skip",
            "A_Dense2",
            "A_LN2",
            "A_Act2",
            "A_Head",
        ]
        critic_nodes = [
            "C_Emb",
            "C_LN1",
            "C_Act1",
            "C_Concat",
            "C_RTU",
            "C_Skip",
            "C_Dense2",
            "C_LN2",
            "C_Act2",
            "C_Head",
        ]
    else:
        nodes["A_Act1"] = {"label": "Tanh", "type": "styleNorm"}
        nodes["C_Act1"] = {"label": "Tanh", "type": "styleNorm"}
        edges += [
            ("A_Emb", "A_Act1"),
            ("A_Act1", "A_Concat"),
            ("A_Dense2", "A_Act2"),
            ("A_Act2", "A_Head"),
            ("C_Emb", "C_Act1"),
            ("C_Act1", "C_Concat"),
            ("C_Dense2", "C_Act2"),
            ("C_Act2", "C_Head"),
        ]
        actor_nodes = [
            "A_Emb",
            "A_Act1",
            "A_Concat",
            "A_RTU",
            "A_Skip",
            "A_Dense2",
            "A_Act2",
            "A_Head",
        ]
        critic_nodes = [
            "C_Emb",
            "C_Act1",
            "C_Concat",
            "C_RTU",
            "C_Skip",
            "C_Dense2",
            "C_Act2",
            "C_Head",
        ]

    return {
        "nodes": nodes,
        "edges": edges,
        "subgraphs": [
            {"id": "Actor", "label": "Actor", "nodes": actor_nodes},
            {"id": "Critic", "label": "Critic", "nodes": critic_nodes},
        ],
        "invisible_edges": [["Actor", "Critic"]],
    }


# ── Param extraction ──────────────────────────────────────────────────────────


def get_dqn_data(config_name, is_drqn=False):
    path = f"{BASE}/{config_name}.json"
    exp = load_exp(path)
    hypers = exp.get_hypers(0)
    rep = hypers["representation"]

    scalar_features = rep.get("scalar_features", ["last_action", "last_reward"])
    if scalar_features and isinstance(scalar_features[0], list):
        scalar_features = scalar_features[0]
    scalars_size = (
        (ACTIONS if "last_action" in scalar_features else 0)
        + (1 if "last_reward" in scalar_features else 0)
        + (1 if "reward_trace" in scalar_features else 0)
    )
    rep["scalars"] = scalars_size

    key = jax.random.PRNGKey(0)
    builder = NetworkBuilder(OBS_SHAPE, rep, key)
    builder.addHead(lambda: hk.Linear(ACTIONS, name="q"), name="q")
    all_params = builder.getParams()

    flat = {}

    def _walk(d, prefix=""):
        for k, v in d.items():
            kp = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                _walk(v, kp)
            else:
                mod = kp.rsplit("/", 1)[0] if "/" in kp else kp
                flat[mod] = flat.get(mod, 0) + int(v.size)

    _walk(all_params)

    conv_out = FOV * FOV * 16
    d = {
        "variant": config_name,
        "fov": FOV,
        "total_params": count_params(all_params),
        "obs_shape": f"{FOV}×{FOV}×3",
        "scalars": scalars_size,
        "has_rt": "reward_trace" in scalar_features,
        "use_layernorm": rep.get("use_layernorm", False),
        "conv_out": conv_out,
        "concat_size": conv_out + scalars_size,
        "modules": flat,
    }
    if is_drqn:
        d["pre_gru_layers"] = rep.get("pre_gru_layers", 0)
        d["post_gru_layers"] = rep.get("post_gru_layers", 0)
    return d


def get_ppo_data(config_name):
    path = f"{BASE}/{config_name}.json"
    with open(path) as f:
        config = json.load(f)
    agent_name = config["agent"]
    meta = config.get("metaParameters", {})
    rep = meta.get("representation", {})
    d_hidden = rep.get("d_hidden", 192)
    hidden_size = rep.get("hidden", 64)
    use_rt = rep.get("use_reward_trace", False)
    use_ln = rep.get("use_layernorm", False)
    batch = 1

    obs = (
        jnp.zeros((batch, FOV, FOV, 3)),
        jnp.zeros((batch, ACTIONS)),
        jnp.zeros((batch, 1)),
        jnp.zeros((batch, 1)),
        jnp.zeros((batch, 1)),
        jnp.zeros((batch, 1)),
    )
    Agent = getPPOAgent(agent_name)
    model = Agent(
        action_dim=ACTIONS,
        d_hidden=d_hidden,
        hidden_size=hidden_size,
        activation=meta.get("activation", "tanh"),
        use_sinusoidal_encoding=rep.get("use_sinusoidal_encoding", False),
        use_reward_trace=use_rt,
        use_layernorm=use_ln,
    )
    key = jax.random.PRNGKey(0)
    hidden_init = (
        model.initialize_memory(batch, d_hidden, hidden_size)
        if "RTU" in agent_name
        else None
    )
    params = model.init(key, hidden_init, obs)
    by_module = params_by_module(params)
    reward_extra = 1 if use_rt else 0
    obs_flat = FOV * FOV * 3
    obs_hidden = hidden_size - ACTIONS - 1 - reward_extra

    return {
        "variant": config_name,
        "fov": FOV,
        "total_params": count_params(params),
        "obs_shape": f"{FOV}×{FOV}×3",
        "obs_flat": obs_flat,
        "obs_hidden": obs_hidden,
        "concat_size": hidden_size,
        "d_hidden": d_hidden,
        "hidden_size": hidden_size,
        "use_layernorm": use_ln,
        "use_reward_trace": use_rt,
        "is_rtu": "RTU" in agent_name,
        "modules": by_module,
    }


# ── Build all variants ────────────────────────────────────────────────────────

META = [
    (
        "DQN_LN_RT",
        lambda n: get_dqn_data(n),
        build_dqn_graph,
        "DQN_LN_RT",
        "DQN with LayerNorm and Reward Trace.",
    ),
    (
        "DQN_LN",
        lambda n: get_dqn_data(n),
        build_dqn_graph,
        "DQN_LN",
        "DQN with LayerNorm.",
    ),
    (
        "DQN",
        lambda n: get_dqn_data(n),
        build_dqn_graph,
        "DQN",
        "Baseline DQN without LayerNorm.",
    ),
    (
        "DRQN_LN_0_2",
        lambda n: get_dqn_data(n, True),
        build_drqn_graph,
        "DRQN_LN_0_2",
        "Recurrent DQN with LayerNorm, 0 pre-GRU layers.",
    ),
    (
        "DRQN_LN_1_1",
        lambda n: get_dqn_data(n, True),
        build_drqn_graph,
        "DRQN_LN_1_1",
        "Recurrent DQN with LayerNorm, 1 pre-GRU layer.",
    ),
    (
        "DRQN_0_2",
        lambda n: get_dqn_data(n, True),
        build_drqn_graph,
        "DRQN_0_2",
        "Recurrent DQN without LayerNorm, 0 pre-GRU layers.",
    ),
    (
        "DRQN_1_1",
        lambda n: get_dqn_data(n, True),
        build_drqn_graph,
        "DRQN_1_1",
        "Recurrent DQN without LayerNorm, 1 pre-GRU layer.",
    ),
    (
        "PPO_LN_RT_128",
        get_ppo_data,
        build_ppo_graph,
        "PPO_LN_RT_128",
        "PPO with LayerNorm and Reward Trace.",
    ),
    ("PPO_LN_128", get_ppo_data, build_ppo_graph, "PPO_LN_128", "PPO with LayerNorm."),
    (
        "PPO_128",
        get_ppo_data,
        build_ppo_graph,
        "PPO_128",
        "Baseline PPO without LayerNorm.",
    ),
    (
        "PPO-RTU_LN_128_512",
        get_ppo_data,
        build_ppo_rtu_graph,
        "PPO-RTU_LN_128_512",
        "RTU PPO with LayerNorm, d_hidden=512.",
    ),
    (
        "PPO-RTU_LN_128",
        get_ppo_data,
        build_ppo_rtu_graph,
        "PPO-RTU_LN_128",
        "RTU PPO with LayerNorm, d_hidden=192.",
    ),
    (
        "PPO-RTU_128_512",
        get_ppo_data,
        build_ppo_rtu_graph,
        "PPO-RTU_128",
        "RTU PPO without LayerNorm, d_hidden=512.",
    ),
    (
        "PPO-RTU_128",
        get_ppo_data,
        build_ppo_rtu_graph,
        "PPO-RTU_128",
        "RTU PPO without LayerNorm, d_hidden=192.",
    ),
]

arch_data = {}
for config_name, data_fn, graph_fn, title, desc in META:
    print(f"Processing {config_name}...")
    try:
        data = data_fn(config_name)
        data["title"] = title
        data["description"] = desc
        graph = graph_fn(data)
        data["graph"] = graph
        data["mermaid_source"] = graph_to_mermaid(graph)
        arch_data[config_name] = data
    except Exception as e:
        import traceback

        print(f"  ERROR {config_name}: {e}")
        traceback.print_exc()

out_path = "public/arch_data.json"
with open(out_path, "w") as f:
    json.dump(arch_data, f, indent=2)

print(f"\nExported {len(arch_data)} variants → {out_path}")
print(f"\n{'Variant':<25}  {'Params':>12}")
print("-" * 40)
for name, d in arch_data.items():
    print(f"  {name:<23}  {d['total_params']:>12,}")
