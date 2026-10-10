import numpy as np

from minimal_mjx.utils import EnvState


def _state():
    return EnvState(
        data='data', obs=np.zeros(3), reward=1.0, done=False,
        metrics={'m': 1}, info={'i': 2},
    )


def test_replace_changes_only_given_fields():
    s = _state()
    r = s.replace(reward=5.0, done=True)
    assert r.reward == 5.0 and r.done is True
    assert r.data is s.data
    assert r.obs is s.obs
    assert r.metrics is s.metrics
    assert r.info is s.info


def test_replace_does_not_mutate_original():
    s = _state()
    s.replace(reward=5.0, info={})
    assert s.reward == 1.0
    assert s.info == {'i': 2}


def test_replace_returns_new_envstate():
    s = _state()
    r = s.replace()
    assert isinstance(r, EnvState)
    assert r is not s
