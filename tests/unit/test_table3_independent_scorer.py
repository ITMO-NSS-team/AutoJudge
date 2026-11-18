"""Check exact attribution semantics independently of the experiment runner."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('table3_independent',Path(__file__).resolve().parents[2]/'rebuttal/table3_reconstruction/verify.py')
checker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_step_is_independent_of_agent_and_converts_zero_based_reference():
    assert checker.score({'agent':'Wrong','step':4},{'agent':'Right','step_zero_based':3})==(0,1,0)
    assert checker.score({'agent':'Right','step':3},{'agent':'Right','step_zero_based':3})==(1,0,0)


def test_agent_matching_is_canonical_and_step_must_be_an_integer():
    reference={'agent':'Worker_A','step_zero_based':0}
    assert checker.score({'agent':'worker_a','step':1},reference)==(0,1,0)
    assert checker.score({'agent':'Worker_A ','step':1},reference)==(0,1,0)
    assert checker.score({'agent':'Worker_A','step':True},reference)==(1,0,0)
    assert checker.score({'agent':'Worker_A','step':1},reference)==(1,1,1)
