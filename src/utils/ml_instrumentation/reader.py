
from pathlib import Path
from typing import Any

import connectorx as cx

import utils.ml_instrumentation._utils.sqlite as sqlu


def get_run_ids(db_path: str | Path, params: dict[str, Any]):
    constraints = ' AND '.join(
        f'[{k}]={sqlu.maybe_quote(v)}' for k, v in params.items()
    )

    query = f'SELECT id FROM _metadata_ WHERE {constraints}'
    return cx.read_sql(f'sqlite://{db_path}', query, return_type='polars')['id'].to_list()
