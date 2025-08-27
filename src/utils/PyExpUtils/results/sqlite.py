import os
import sqlite3
import logging
import pandas as pd
import PyExpUtils.results.sqlite_utils as sqlu

from filelock import FileLock
from typing import Iterable, Sequence

from PyExpUtils.collection.Collector import Collector
from PyExpUtils.models.ExperimentDescription import ExperimentDescription
from PyExpUtils.results.indices import listIndices
from PyExpUtils.results.migrations import maybe_migrate
from PyExpUtils.results.tools import getHeader, getParamValues
from PyExpUtils.results._utils.shared import hash_values
from PyExpUtils.results.sqlite import get_cid



def get_cid(cur: sqlite3.Cursor, header: Sequence[str], exp: ExperimentDescription, idx: int) -> int:
    values = getParamValues(exp, idx, header)

    # first see if a cid already exists
    if len(header) > 0:
        c = sqlu.constraints_from_lists(header, values)
        res = cur.execute(f'SELECT config_id FROM hyperparameters WHERE {c}')
    else:
        res = cur.execute('SELECT config_id FROM hyperparameters')

    cids = res.fetchall()
    if len(cids) > 0:
        return cids[0][0]

    # otherwise create and store a cid
    cid = hash_values(values)

    if len(header) > 0:
        c_str = ','.join(map(sqlu.maybe_quote, header))
        v_str = ','.join(map(str, map(sqlu.maybe_quote, values)))
        cur.execute(f'INSERT INTO hyperparameters({c_str},config_id) VALUES({v_str},{cid})')
    else:
        cur.execute(f'INSERT INTO hyperparameters(config_id) VALUES({cid})')

    return cid


def detectMissingIndices(exp: ExperimentDescription, runs: int, base: str = './'): # noqa: C901
    context = exp.buildSaveContext(0, base=base)
    nperms = exp.numPermutations()

    header = getHeader(exp)

    # first case: no data
    if not context.exists('results.db'):
        yield from listIndices(exp, runs)
        return

    db_file = context.resolve('results.db')
    con = sqlite3.connect(db_file, timeout=30)
    cur = con.cursor()

    tables = sqlu.get_tables(cur)
    if '_metadata_' not in tables:
        yield from listIndices(exp, runs)
        con.close()
        return

    # TODO: make robust to changes in configs.
    expected_seeds = set(range(nperms * runs))
    # rows = cur.execute('SELECT DISTINCT id FROM _metadata_').fetchall()
    rows = cur.execute('SELECT DISTINCT id from mean_ewm_reward').fetchall()
    seeds = set(d[0] for d in rows)
    needed = expected_seeds - seeds
    for seed in needed:
        yield seed

    con.close()
