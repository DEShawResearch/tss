#!/usr/bin/env python

import tss
import numpy as np

class Gaussians:    
    def __init__(self):
        self.state = 0
        self.deviation = 1

    def sample(self, rung):
        self.state = np.random.normal(loc = rung, scale = self.deviation)

    def energy(self, N):
        return ((self.state - np.arange(N)) / self.deviation)**2 / 2


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--rung-count', type=int, default=10)
    parser.add_argument('-W', '--window-size', type=int, default=10)
    parser.add_argument('-t', '--timesteps', type=int, default=1000)
    args = parser.parse_args()
    N = args.rung_count
    window_size = args.window_size
    Ntimesteps = args.timesteps
    gb = tss.GraphBuilder()
    edge_id = gb.add_edge(['left', 'right'], [N], [window_size])
    gb.build()
    graph_ark = gb.cerealize()

    estimator = tss.Sampler(graph_ark)
    estimator.initDummyModel()
    g = Gaussians()

    for i in range(Ntimesteps):
        current_rung = estimator.getReplicaRung(0)
        g.sample(current_rung)
        estimator.setEnergies(0, g.energy(N))
        estimator.step()
    estimator.updateReporting()
    free_energy = estimator.getFreeEnergies()
    print(free_energy)
        
