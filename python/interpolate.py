''' interpolate.py
    interpolation methods used in TSS graph construction
'''
import numpy as np

def interpolate_linear(bounds, number):
    """Produce number rungs linearly interpolated between the two bounds"""
    return np.linspace(bounds[0], bounds[1], number)

def interpolate_piecewise(bounds, number):
    """Produce number rungs piecewise linearly interpolated between the sequence of bound points,
        with an equal number of rungs allocated to each segment
        
        >>> interpolate_piecewise([0 10 90], 10)
            np.array([0, 2.5, 5.0, 7.5, 10, 10, 30, 50, 70, 90])"""

    nsegments = len(bounds) - 1
    assert(nsegments >= 2), "must have at least two points for piecewise linear interpolation"
    assert(number % nsegments == 0), "number of interpolation segments must evenly divide number of rungs"

    nrungs_per_segment = number // nsegments

    return np.concatenate([np.linspace(bounds[k], bounds[k+1], nrungs_per_segment) for k in range(len(bounds) - 1)])

def interpolate_polynomial(bounds, number):
    power = bounds[2]
    return np.power(interpolate_linear(bounds[0:2], number), power)

def interpolate_geometric(bounds, number):
    """Produce number rungs geometrically interpolated between the two bounds"""
    assert (len(bounds) == 2), "bounds must be length 2 for geometric interpolation"

    ratio = np.power(float(bounds[1]) / bounds[0], 1. / (number - 1))
    powers = np.power(ratio, np.arange(number))
    return bounds[0] * powers

def interpolate_geometric_increment(bounds, number):
    """Produce number rungs interpolated between the two bounds such that
        the ratio of adjacent differences is constant. The bias controls
        in which direction the rungs are most finely spaced."""
    assert (len(bounds) == 3), "bounds must be length 3 including bias for geometric increment interpolation"
    
    bias = bounds[2]

    # The bias is interpreted in the same way that Duluxan's code interprets it. 
    delta = bounds[1] - bounds[0]
    theta = np.power(bias, 1. / (number - 2))
    theta_powers = np.power(theta, np.arange(number - 1))
    theta_seq = np.cumsum(theta_powers)

    (theta_seq, theta_divisor) = (theta_seq[:-1], theta_seq[-1])

    result = np.concatenate([[bounds[0]],
                            (bounds[0] + delta * theta_seq / theta_divisor),
                            [bounds[1]]])
    return result
    
def interpolate_explicit(bounds, number):
    """The bounds specify the entire interpolation schedule."""
    assert (len(bounds) == number), "bounds must be length n for explicit value specification"
    return bounds
