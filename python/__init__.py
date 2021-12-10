import _tss
from .tss_graph_builder import *
from .tss_context_manager import *
from ark import _ark

class Sampler(_tss.Sampler):
    def __init__(self, graph_ark, **kwargs):
        if isinstance(graph_ark, dict):
            graph_ark = _ark.from_object(graph_ark)
        super().__init__(graph_ark, **kwargs)

__all__ = ['Sampler', 'GraphBuilder', 'Gaussian', 'delta_free_energies']
