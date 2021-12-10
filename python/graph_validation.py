'''graph_validation.py
    Methods for testing the sanity of a serialized TSS graph ark.
'''
from collections import Counter
from .util import nested_flatten

def check_two_windows_every_rung(cerealized):
    c = Counter()
    for w in cerealized['graph']['windows']:
        c.update(Counter(nested_flatten(w['rung_set'])))

    assert(set(c.keys()) == set(range(len(cerealized['graph']['rungs'])))), "Bad rung IDs in window output"

    exceptions = {k: v for k, v in c.items() if v != 2}
    if (len(exceptions) != 0):
        warnings.warn("All rungs should be in exactly 2 windows, exceptions: {}. If using primary_window_tiling_only, you may disregard this message.".format(exceptions), RuntimeWarning)
