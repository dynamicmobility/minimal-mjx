import pytest
from ml_collections import config_dict

from minimal_mjx.utils.logging import (
    RUN_ID_FNAME, flatten_config, load_run_id, save_run_id, unflatten_config,
)


def test_flatten_config_dict():
    nested = {'a': 1, 'b': {'c': 2, 'd': {'e': [1, 2]}}}
    assert flatten_config(nested) == {'a': 1, 'b/c': 2, 'b/d/e': [1, 2]}


def test_flatten_config_configdict():
    cfg = config_dict.ConfigDict({'a': {'b': 1}, 'c': 'x'})
    assert flatten_config(cfg) == {'a/b': 1, 'c': 'x'}


def test_flatten_config_separator():
    assert flatten_config({'a': {'b': 1}}, sep='.') == {'a.b': 1}


def test_flatten_config_rejects_separator_in_key():
    with pytest.raises(ValueError, match='reserved'):
        flatten_config({'a': {'b/c': 1}})


def test_unflatten_round_trip():
    nested = {'a': 1, 'b': {'c': 2, 'd': {'e': [1, 2]}}}
    assert unflatten_config(flatten_config(nested)) == nested


def test_unflatten_skips_private_keys():
    assert unflatten_config({'_wandb': {}, 'a/b': 1}) == {'a': {'b': 1}}


def test_unflatten_conflict_raises():
    with pytest.raises(ValueError, match='not a section'):
        unflatten_config({'a': 1, 'a/b': 2})


def test_run_id_round_trip(tmp_path):
    assert load_run_id(tmp_path) is None
    save_run_id(tmp_path, None)
    assert not (tmp_path / RUN_ID_FNAME).exists()

    class Run:
        id = 'abc123'

    save_run_id(tmp_path, Run())
    assert load_run_id(tmp_path) == 'abc123'
