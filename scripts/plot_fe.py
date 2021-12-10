#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
from msys.molfile import DtrReader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('etr', help='The TSS etr')
args = parser.parse_args()

# get FEs
e = DtrReader(args.etr)
kv = {}
e.frame(e.nframes - 1, keyvals=kv)

# plot relative to first rung FE
plt.plot(kv['TSS_FE'] - kv['TSS_FE'][0])
plt.show()
