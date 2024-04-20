import numpy as np
from typing import NamedTuple
class CameraInfo4K4D(NamedTuple):
    R: np.array
    T: np.array
    K: np.array
    invK: np.array
    H: int
    W: int
    C: np.array
    RT: np.array
    Rvec: np.array
    P: np.array
    D: np.array
    t: float
    v: float
    n: float
    f: float
    bounds: np.array
    ccm: np.array