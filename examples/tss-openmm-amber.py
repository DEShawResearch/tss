#!/usr/bin/env python

from simtk import openmm as mm
from simtk.openmm import app
from simtk.unit import *
import os
import numpy as np

################################################################################

class Simulation(object):
    def __init__(self, prm_file, crd_file, temperature):
        prmtop = app.AmberPrmtopFile(prm_file)
        inpcrd = app.AmberInpcrdFile(crd_file)

        self.system = prmtop.createSystem(nonbondedMethod=app.PME,
                                          nonbondedCutoff=12*angstrom,
                                          switchDistance=10*angstrom,
                                          removeCMMotion=True,
                                          ewaldErrorTolerance=0.0001,
                                          constraints=app.HBonds)

        self.barostat = mm.MonteCarloBarostat(1*bar, temperature)
        self.system.addForce(self.barostat)

        self.integrator = mm.LangevinIntegrator(temperature, 1/picosecond,
                                                0.002*picoseconds)
        platform = mm.Platform.getPlatformByName('CUDA')

        self.simulation = app.Simulation(
            prmtop.topology, self.system, self.integrator, platform)
        self.simulation.context.setPositions(inpcrd.positions)
        self.simulation.context.computeVirtualSites()
        self.current_temperature = temperature

    def change_temperature(self, new_temperature):
        scale = new_temperature / self.current_temperature
        self.integrator.setTemperature(new_temperature)
        self.simulation.context.setVelocitiesToTemperature(new_temperature)
        self.simulation.context.setParameter(mm.MonteCarloBarostat.Temperature(),
                                             new_temperature)
        self.current_temperature = new_temperature

    def propagate(self, time_steps=2000): # run for 2ps
        self.simulation.step(time_steps)

    def snapshot(self):
        state = self.simulation.context.getState(enforcePeriodicBox=True,
                                                 getEnergy=True,
                                                 getVelocities=True,
                                                 getPositions=True)
        box = state.getPeriodicBoxVectors()
        cell = np.array([[float(box[0][0].value_in_unit(angstrom)), 0, 0],
                              [0, float(box[1][1].value_in_unit(angstrom)), 0],
                              [0, 0, float(box[2][2].value_in_unit(angstrom))]])
        return state.getPotentialEnergy(), \
            state.getPositions(asNumpy=True).value_in_unit(angstrom), \
            cell

import tss

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prm-file', type=str)
    parser.add_argument('--crd-file', type=str)
    args = parser.parse_args()
    sim = Simulation(prm_file=args.prm_file, crd_file=args.crd_file,
                     temperature=273 * kelvin)
    Nrungs = 100
    window_size = 50
    print('Window size', window_size)
    gb = tss.GraphBuilder()
    edge_id = gb.add_edge(['cold', 'hot'], [Nrungs], [window_size])
    gb.build()
    graph_ark = gb.cerealize()

    estimator = tss.Sampler(graph_ark)
    estimator.initDummyModel()
    temperatures = np.linspace(273, 400, Nrungs)
    betas = 1 / ((temperatures * kelvin * MOLAR_GAS_CONSTANT_R).in_units_of(
        kilocalories/mole))
    Ntimesteps = int(5 * 1000000 / 2) # 1us / 5ps

    sim.simulation.reporters.append(app.StateDataReporter(sys.stdout,
        25000, step=True, kineticEnergy=True, potentialEnergy=True,
        temperature=True, volume=True, speed=True))

    frames_per_file = 100
    file_count = 0
    write_every = 2
    max_rung = 0

    trj_frames = []
    trj_boxes = []
    trj_temperatures = []
    trj_rung = []
    trj_fe = []

    for i in range(Ntimesteps):
        current_rung = estimator.getReplicaRung(0)
        current_temperature = temperatures[current_rung]
        sim.change_temperature(current_temperature * kelvin)
        sim.propagate()

        energy, frame, box = sim.snapshot()

        trj_rung.append(current_rung)

        energy = energy.in_units_of(kilocalories/mole) * betas
        estimator.setEnergies(0, energy)
        estimator.step()
        free_energy = estimator.getFreeEnergies()

        if (i % write_every) == 0:
            trj_frames.append(frame)
            trj_boxes.append(box)
            trj_temperatures.append(current_temperature)
            trj_fe.append(free_energy)

        if len(trj_temperatures) == frames_per_file:
            np.savez_compressed('trj_frames_%04i'%file_count,
                frames=trj_frames, cells=trj_boxes,
                temperatures=trj_temperatures,
                fe=trj_fe)
            trj_frames = []
            trj_boxes = []
            trj_temperatures = []
            trj_fe = []
            file_count = file_count + 1
            max_rung = max(max_rung, max(trj_rung))
            print('.'*80)
            print('Dumped block', file_count - 1)
            print('Exploration:', (100*max_rung) / Nrungs, '%')
            trj_rung = []

