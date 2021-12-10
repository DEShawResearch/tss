import numpy as np
import pytest

@pytest.fixture
def basic_graph():
    import tss
    gb = tss.GraphBuilder()
    rung_count = 32
    window_size = int(rung_count)
    edge = gb.add_edge(['A', 'B'], [rung_count], [window_size])
    gb.add_schedule(edge, 'mean_0', 'linear', [1, rung_count])
    return gb.build()

@pytest.fixture
def deviation_graph():
    import tss
    gb = tss.GraphBuilder()
    rung_count = 32
    window_size = int(rung_count)
    edge = gb.add_edge(['A', 'B'], [rung_count], [window_size])
    gb.add_schedule(edge, 'deviation_0', 'linear', [1, rung_count])
    return gb.build()

def test_import():
    import tss

def test_create_sampler(basic_graph):
    import tss, ark
    s = tss.Sampler(basic_graph)

def test_rung_window_move(basic_graph):
    import tss

    initial_rung = 0
    s = tss.Sampler(basic_graph)
    s.initDummyModel(1)
    s.setReplicaRung(0, initial_rung)

    energies = np.zeros(s.getNumRungs())
    s.setEnergies(0, energies)
    initial_window = s.getReplicaWindow(0)

    # step with antithetic move
    s.step()
    s.step()
    s.step()
    s.step()
    s.step()
    assert(s.getReplicaWindow(0) != initial_window)
    assert(s.getReplicaRung(0) != initial_rung)

# ensure that all rungs of graph are visited in a reasonable time
def test_rung_coverage(basic_graph):
    import tss

    initial_rung = 0
    s = tss.Sampler(basic_graph)
    s.initGaussiansModel()
    s.setReplicaRung(0, initial_rung)

    nsteps = 1000
    rung_counts = np.zeros(32)
    for i in range(nsteps):
        s.step()
        rung_counts[s.getReplicaRung(0)] += 1
    assert((rung_counts > 0).all())

# test running with coordinate invariance and converging tempering of sigma
def test_coordinate_invariance(deviation_graph):
    import tss

    initial_rung = 0
    s = tss.Sampler(deviation_graph)
    s.initGaussiansModel()
    s.setReplicaRung(0, initial_rung)

    nsteps = 100000
    rung_counts = np.zeros(32)
    s.step(nsteps)

    # make sure our absolute convergence is reasonable
    s.updateReporting()
    fes = s.getFreeEnergies()
    print('deviation is', fes[-1] - fes[0])
    assert(abs(fes[-1] - fes[0] + np.log(32/1)) < 0.1)

# make sure that estimated error decreases appropriately
def test_rung_coverage(basic_graph):
    import tss

    initial_rung = 0
    s = tss.Sampler(basic_graph, visit_control_eta=4)
    s.initGaussiansModel()
    s.setReplicaRung(0, initial_rung)

    nsteps = 2**17
    next_exponent = 11
    errors = []
    for i in range(nsteps):
        s.step()
        if i > 2**next_exponent:
            err = s.getPairErrors()[1][0]
            errors.append(err)
            next_exponent += 0.5

    # make sure our absolute convergence is reasonable
    s.updateReporting()
    fes = s.getFreeEnergies()
    assert(abs(fes[-1] - fes[0]) < 0.5)
    
    log_errors = np.log(errors)
    print('Errors:', errors)
    from scipy.stats import linregress
    slope = linregress(range(len(log_errors)), log_errors).slope
    print('slope', slope)
    assert(np.exp(4*slope) < 0.6) # on average, running 4x longer should reduce error by a factor of 2.  Giving a bit of leeway.

