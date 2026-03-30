"""
Paper B Core Simulation Engine
===============================
Shared by all phases: damage dynamics, survival models, likelihood, fitting.
"""
import numpy as np
from scipy.optimize import minimize

# ── Global defaults ──────────────────────────────────────────────────────────
DT = 0.1          # integration time step (days)
NQ = 8            # quadrature points for tolerance distribution
PQ = np.linspace(0.12, 0.88, NQ)
DPQ = np.diff(PQ, prepend=0)

# ── Damage dynamics ──────────────────────────────────────────────────────────

def sim_damage_1c(c_ext_func, kd, tmax=4.2):
    """GUTS-RED 1-compartment damage: dD/dt = kd*(Cext - D).
    
    At steady state, D_ss = C_ext.  This is the standard GUTS-RED
    "scaled internal concentration" formulation.
    
    Args:
        c_ext_func: callable(t) -> external concentration, or float for constant
        kd: dominant rate constant (d^-1)
        tmax: simulation end time
    
    Returns:
        t, D arrays
    """
    if isinstance(c_ext_func, (int, float)):
        _c = float(c_ext_func)
        c_ext_func = lambda t: _c
    
    n = int(tmax / DT) + 1
    t = np.linspace(0, tmax, n)
    D = np.zeros(n)
    
    for i in range(1, n):
        cext = c_ext_func(t[i-1])
        D[i] = max(D[i-1] + DT * kd * (cext - D[i-1]), 0.0)
    
    return t, D


def sim_damage_1c_pulsed(pulse_profile, kd, tmax=None):
    """GUTS-RED 1-compartment damage under a pulsed exposure profile.
    
    Args:
        pulse_profile: list of (t_start, t_end, concentration) tuples
        kd: dominant rate constant
        tmax: simulation end time (auto-determined if None)
    
    Returns:
        t, D arrays
    """
    if tmax is None:
        tmax = max(p[1] for p in pulse_profile) + 5.0
    
    def c_ext_func(t):
        for ts, te, conc in pulse_profile:
            if ts <= t < te:
                return conc
        return 0.0
    
    return sim_damage_1c(c_ext_func, kd, tmax)


# ── Tolerance distribution ───────────────────────────────────────────────────

def log_logistic_quantiles(median, beta, pq=PQ):
    """Log-logistic quantile thresholds."""
    return median * (pq / (1 - pq)) ** (1.0 / beta)


# ── Survival models ──────────────────────────────────────────────────────────

def survival_classB(D, t, k, th_med, beta, hb=0.001):
    """Class B (peak-based) population survival.
    
    H_B = k * [max_D - theta]^+ + hb*t
    """
    thresholds = log_logistic_quantiles(th_med, beta)
    M = np.maximum.accumulate(D)  # running maximum
    H = k * np.maximum(M[:, None] - thresholds[None, :], 0) + hb * t[:, None]
    S_ind = np.exp(-H)
    S = np.clip((S_ind * DPQ[None, :]).sum(axis=1), 1e-15, 1.0)
    return S


def survival_SD(D, t, bw, z_med, beta, hb=0.001):
    """GUTS-SD (Class A, cumulative hazard) population survival.
    
    H_A = integral of bw*[D - z]^+ dt + hb*t
    """
    thresholds = log_logistic_quantiles(z_med, beta)
    exc = np.maximum(D[:, None] - thresholds[None, :], 0)
    H_cum = bw * np.cumsum(exc, axis=0) * DT + hb * t[:, None]
    S_ind = np.exp(-H_cum)
    S = np.clip((S_ind * DPQ[None, :]).sum(axis=1), 1e-15, 1.0)
    return S


# ── Likelihood ───────────────────────────────────────────────────────────────

def neg_log_likelihood(obs_t, obs_alive, pred_t, pred_S):
    """Conditional binomial NLL.
    
    Args:
        obs_t: observation times
        obs_alive: number alive at each time
        pred_t: simulation time array
        pred_S: predicted survival fraction array
    
    Returns:
        NLL (scalar)
    """
    S_pred = np.interp(obs_t, pred_t, pred_S)
    S_pred = np.clip(S_pred, 1e-10, 1 - 1e-10)
    
    nll = 0.0
    for j in range(len(obs_t) - 1):
        a_j = obs_alive[j]
        a_next = obs_alive[j + 1]
        q = np.clip(S_pred[j + 1] / S_pred[j], 1e-10, 1 - 1e-10)
        nll -= a_next * np.log(q) + (a_j - a_next) * np.log(1 - q)
    
    return nll


def total_nll_multi_conc(datasets, pred_func):
    """Total NLL across multiple concentrations.
    
    Args:
        datasets: list of (conc, obs_t, obs_alive) tuples
        pred_func: callable(conc) -> (pred_t, pred_S)
    """
    total = 0.0
    for conc, obs_t, obs_alive in datasets:
        pred_t, pred_S = pred_func(conc)
        total += neg_log_likelihood(obs_t, obs_alive, pred_t, pred_S)
    return total


# ── Data generation ──────────────────────────────────────────────────────────

def generate_lc50_data(concs, true_params, n_per_conc=20, obs_times=None, rng=None):
    """Generate synthetic 96h LC50 dataset from Class B model (GUTS-RED).
    
    Args:
        concs: array of external concentrations
        true_params: dict with keys: kd, k, th_med, beta, hb
        n_per_conc: organisms per concentration
        obs_times: observation times (default: [0,1,2,3,4])
        rng: numpy random generator
    
    Returns:
        list of (conc, obs_times, alive_counts)
    """
    if obs_times is None:
        obs_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    if rng is None:
        rng = np.random.default_rng()
    
    p = true_params
    datasets = []
    
    for c in concs:
        t, D = sim_damage_1c(c, p['kd'], tmax=obs_times[-1] + 0.2)
        S = survival_classB(D, t, p['k'], p['th_med'], p['beta'], p['hb'])
        
        S_obs = np.interp(obs_times, t, S)
        
        alive = [n_per_conc]
        for j in range(1, len(obs_times)):
            q = np.clip(S_obs[j] / S_obs[j-1], 1e-10, 1 - 1e-10)
            alive.append(rng.binomial(alive[-1], q))
        
        datasets.append((c, obs_times, np.array(alive)))
    
    return datasets


def design_concentrations(th_med, n_concs=7):
    """Auto-design concentration range spanning sub-threshold to 2×LC50.
    
    Under GUTS-RED, D_ss = C_ext, so threshold ≈ LC50.
    """
    c_min = 0.1 * th_med
    c_max = 3.0 * th_med
    return np.geomspace(c_min, c_max, n_concs)


# ── Model fitting ────────────────────────────────────────────────────────────

def fit_classB(datasets, kd, hb=0.001, n_starts=10, 
               fit_kd=False, rng=None):
    """Fit Class B model to LC50 datasets (log-transformed optimisation).
    
    Args:
        datasets: from generate_lc50_data
        kd: dominant rate (fixed or initial if fit_kd=True)
        fit_kd: if True, kd is a free parameter
        n_starts: number of random starting points
        rng: random generator
    
    Returns:
        dict with fitted params and NLL
    """
    if rng is None:
        rng = np.random.default_rng()
    
    def objective(log_p):
        if fit_kd:
            kd_val = 10**log_p[0]
            k_val = 10**log_p[1]
            th_val = 10**log_p[2]
        else:
            kd_val = kd
            k_val = 10**log_p[0]
            th_val = 10**log_p[1]
        
        nll = 0.0
        for conc, obs_t, obs_alive in datasets:
            t_sim, D = sim_damage_1c(conc, kd_val, tmax=obs_t[-1] + 0.2)
            S = survival_classB(D, t_sim, k_val, th_val, 5.0, hb)
            nll += neg_log_likelihood(obs_t, obs_alive, t_sim, S)
        
        if np.isnan(nll) or np.isinf(nll):
            return 1e12
        return nll
    
    best = None
    for _ in range(n_starts):
        if fit_kd:
            x0 = [rng.uniform(-2, 1),       # log10(kd)
                   rng.uniform(-6, -1),      # log10(k)
                   rng.uniform(-1, 2)]       # log10(th_med)
        else:
            x0 = [rng.uniform(-6, -1),       # log10(k)
                   rng.uniform(-1, 2)]       # log10(th_med)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    
    if best is None:
        return {'nll': 1e12, 'k': np.nan, 'th_med': np.nan, 'kd': kd, 'converged': False}
    
    if fit_kd:
        return {'nll': best.fun, 'kd': 10**best.x[0],
                'k': 10**best.x[1], 'th_med': 10**best.x[2],
                'converged': best.success}
    else:
        return {'nll': best.fun, 'kd': kd,
                'k': 10**best.x[0], 'th_med': 10**best.x[1],
                'converged': best.success}


def fit_SD(datasets, kd, hb=0.001, n_starts=10, 
           fit_kd=False, rng=None):
    """Fit GUTS-SD model to LC50 datasets (log-transformed optimisation).
    
    Returns:
        dict with fitted params and NLL
    """
    if rng is None:
        rng = np.random.default_rng()
    
    def objective(log_p):
        if fit_kd:
            kd_val = 10**log_p[0]
            bw_val = 10**log_p[1]
            z_val = 10**log_p[2]
        else:
            kd_val = kd
            bw_val = 10**log_p[0]
            z_val = 10**log_p[1]
        
        nll = 0.0
        for conc, obs_t, obs_alive in datasets:
            t_sim, D = sim_damage_1c(conc, kd_val, tmax=obs_t[-1] + 0.2)
            S = survival_SD(D, t_sim, bw_val, z_val, 5.0, hb)
            nll += neg_log_likelihood(obs_t, obs_alive, t_sim, S)
        
        if np.isnan(nll) or np.isinf(nll):
            return 1e12
        return nll
    
    best = None
    for _ in range(n_starts):
        if fit_kd:
            x0 = [rng.uniform(-2, 1),       # log10(kd)
                   rng.uniform(-6, -1),      # log10(bw)
                   rng.uniform(-1, 2)]       # log10(z_med)
        else:
            x0 = [rng.uniform(-6, -1),       # log10(bw)
                   rng.uniform(-1, 2)]       # log10(z_med)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    
    if best is None:
        return {'nll': 1e12, 'bw': np.nan, 'z_med': np.nan, 'kd': kd, 'converged': False}
    
    if fit_kd:
        return {'nll': best.fun, 'kd': 10**best.x[0],
                'bw': 10**best.x[1], 'z_med': 10**best.x[2],
                'converged': best.success}
    else:
        return {'nll': best.fun, 'kd': kd,
                'bw': 10**best.x[0], 'z_med': 10**best.x[1],
                'converged': best.success}


# ── TWA calculation ──────────────────────────────────────────────────────────

def compute_twa(c_ext_func, tmax, window=7.0, dt=0.1):
    """Compute time-weighted average with moving window.
    
    Returns:
        max TWA over all windows
    """
    t = np.arange(0, tmax, dt)
    c = np.array([c_ext_func(ti) for ti in t])
    
    win_steps = int(window / dt)
    if win_steps >= len(c):
        return np.mean(c)
    
    # Moving average
    cumsum = np.cumsum(c)
    twa = (cumsum[win_steps:] - cumsum[:-win_steps]) / win_steps
    return np.max(twa) if len(twa) > 0 else np.mean(c)


def twa_predict_survival(twa_conc, lc50, slope=3.0):
    """Simple probit prediction from TWA concentration.
    
    Args:
        twa_conc: TWA concentration
        lc50: 96h LC50 value
        slope: probit slope (default 3.0)
    
    Returns:
        predicted survival fraction
    """
    if twa_conc <= 0 or lc50 <= 0:
        return 1.0
    # Log-logistic dose-response
    ratio = twa_conc / lc50
    if ratio <= 0:
        return 1.0
    return 1.0 / (1.0 + ratio**slope)


# ── Pulse profile helpers ────────────────────────────────────────────────────

def make_pulse_profile(n_pulses, pulse_dur, interval, conc):
    """Create a pulse exposure profile.
    
    Args:
        n_pulses: number of pulses
        pulse_dur: duration of each pulse (days)
        interval: interval between pulse starts (days)
        conc: pulse concentration
    
    Returns:
        list of (t_start, t_end, concentration) tuples
    """
    profile = []
    for i in range(n_pulses):
        t_start = i * interval
        t_end = t_start + pulse_dur
        profile.append((t_start, t_end, conc))
    return profile


def profile_to_func(pulse_profile):
    """Convert pulse profile to callable c_ext(t)."""
    def c_ext(t):
        for ts, te, conc in pulse_profile:
            if ts <= t < te:
                return conc
        return 0.0
    return c_ext
