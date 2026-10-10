from ml_collections import config_dict

from minimal_mjx.utils.logging import _flatten_config


def test_flatten_config_dict():
    nested = {'a': 1, 'b': {'c': 2, 'd': {'e': [1, 2]}}}
    assert _flatten_config(nested) == {'a': 1, 'b/c': 2, 'b/d/e': [1, 2]}


def test_flatten_config_configdict():
    cfg = config_dict.ConfigDict({'a': {'b': 1}, 'c': 'x'})
    assert _flatten_config(cfg) == {'a/b': 1, 'c': 'x'}


def test_flatten_config_separator():
    assert _flatten_config({'a': {'b': 1}}, sep='.') == {'a.b': 1}
