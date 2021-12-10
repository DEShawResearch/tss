import os
import numpy as np
import itertools as it
from collections.abc import Iterable

def first_different_index(l1, l2):
    return next(i for i in range(len(l1)) if l1[i] != l2[i])

def merge_dicts(dicts):
    keys = set(it.chain(*[list(d.keys()) for d in dicts]))
    merged = {k: [d.get(k) for d in dicts] for k in keys}

    return merged

def nested_flatten(l):
    for elt in l:
        if isinstance(elt, Iterable):
            for eelt in nested_flatten(elt):
                yield eelt
        else:
            yield elt

def param_equiv(p1, p2, epsilon=1e-10):
    """Given two parameter dictionaries, check that they're equal
        to within floating point tolerance."""
    return set(p1.keys()) == set(p2.keys()) and \
        all([abs(p1[k] - p2[k]) < epsilon for k in p1.keys()])

def tup_remove(tup, i):
    return tup[:i] + tup[i+1:]

def tup_replace(tup, i, x):
    return tup[:i] + (x,) + tup[i+1:]

def tup_split(tup, i):
    return tup[:i], tup[i:]
