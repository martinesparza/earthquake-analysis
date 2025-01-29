from dataclasses import dataclass


@dataclass
class Params:
    BIN_SIZE = 0.03  # Desired bin size for analysis.

@dataclass
class colors:
    M1 = 'r'
    Dls = 'g'
    all = 'k'
    corr_cmap = 'viridis'
