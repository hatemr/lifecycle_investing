import numpy as np


def compute_rwm(p: np.array, 
                fv_0: float, 
                R_A: np.array, 
                W: np.array, 
                funcP, 
                funcW):
    """
    Compute returns using matrices
    """
    
    assert p.ndim==3
    assert R_A.ndim==3
    assert W.ndim==3

    p[0] = fv_0
    
    for t in range(R_A.shape[0]):
        if t>1:
            p[t]=5
            W[t,:] = np.arrany([]) #funcW()