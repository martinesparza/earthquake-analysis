from dataclasses import dataclass


@dataclass
class Params:
    BIN_SIZE = 0.03  # Desired bin size for analysis.
    
@dataclass
class colors:
    M1 = 'tab:blue'
    Dls = 'tab:orange'
    corr_cmap = 'viridis'