import numpy as np
from sksurv.metrics import concordance_index_censored

def compute_c_index(censorship, event_time, risk_score):
    censorship = np.array(censorship).reshape(-1)
    event_time = np.array(event_time).reshape(-1)
    risk_score = np.array(risk_score).reshape(-1)
    
    c_index = concordance_index_censored((1 - censorship).astype(bool), event_time, risk_score, tied_tol=1e-08)[0]
    
    return c_index