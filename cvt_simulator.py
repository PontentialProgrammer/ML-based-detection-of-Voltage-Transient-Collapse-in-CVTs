import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class CVTParams:
    C1: float; C2: float; LC: float; RC: float
    CC: float; CP: float; LM: float; RM: float
    RW: float; f0: float; Vpk: float

# Table I from paper
PROBST = CVTParams(
    C1=12730e-12, C2=83780e-12, LC=69.2, RC=174.5,
    CC=119.3e-12, CP=300e-12, LM=6269.7, RM=1367.1e3,
    RW=690.0, f0=60.0, Vpk=15018.9
)
KOJOVIC_1 = CVTParams(
    C1=5650e-12, C2=81100e-12, LC=58.3, RC=228.0,
    CC=130e-12, CP=140e-12, LM=1000.0, RM=1000e3,
    RW=400.0, f0=60.0, Vpk=7414.6
)
KOJOVIC_3 = CVTParams(
    C1=6750e-12, C2=84110e-12, LC=74.4, RC=120.0,
    CC=1200e-12, CP=400e-12, LM=1600.0, RM=1000e3,
    RW=230.0, f0=60.0, Vpk=8457.5
)


def derived(p: CVTParams):
    CEQ = p.C1 + p.C2
    CT  = p.CP + p.CC
    w0  = 2 * np.pi * p.f0

    # Fast circuit (Appendix B.1-B.4)
    af  = 0.5 * (1 / (CT * (p.RW + p.RM)) + p.RC / p.LC)
    wf  = 1 / np.sqrt(CT * p.LC)
    FCf = (p.RC + p.RW + p.RM) / (p.RW + p.RM)
    wfc = np.sqrt(max(wf**2 * FCf - af**2, 0))

    # Slow circuit (Appendix B.5-B.8)
    as_ = 0.5 * (1 / (CEQ * p.RM) + p.RM / p.LM)
    ws  = 1 / np.sqrt(p.LM * CEQ)
    FCs = 1.0
    disc = ws**2 / FCs - (as_ / FCs)**2
    wsc  = np.sqrt(abs(disc))

    return CEQ, CT, w0, af, wfc, as_, ws, wsc, FCs, disc


def fast_peak(t, p):
    CEQ, CT, w0, af, wfc, *_ = derived(p)
    # Eq. 10
    return (p.Vpk / CT) * np.exp(-af * t) * (
        (1 / wfc) * (af * p.CP + (p.CC / CT) / (p.RC + p.RW + p.RM)) * np.sin(wfc * t)
        + p.CP * np.cos(wfc * t)
    )


def fast_zero(t, p):
    CEQ, CT, w0, af, wfc, *_ = derived(p)
    # Eq. 11
    return p.Vpk * w0 * np.exp(-af * t) * (
        (1 / wfc) * (p.CP / CT - af * p.LC / (p.RC + p.RW + p.RM)) * np.sin(wfc * t)
        - (p.LC / (p.RC + p.RW + p.RM)) * np.cos(wfc * t)
    )


def _slow_response(t, p, A, B, as_, FCs, wsc, disc):
    """Underdamped (oscillatory) or overdamped (two real roots)."""
    if disc >= 0:
        decay = np.exp(-(as_ / FCs) * t)
        return p.LM * decay * (
            -(as_ / FCs * A + wsc * B) * np.sin(wsc * t)
            + (wsc * A - as_ / FCs * B) * np.cos(wsc * t)
        )
    else:
        r1 = -(as_ / FCs) + wsc
        r2 = -(as_ / FCs) - wsc
        C1 = (A * r2 - B) / (r2 - r1)
        C2 = (B - A * r1) / (r2 - r1)
        return p.LM * (C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t))


def slow_peak(t, p):
    CEQ, CT, w0, af, wfc, as_, ws, wsc, FCs, disc = derived(p)
    eps = 1e-10
    # Eq. 17, 18
    A = p.Vpk * (CEQ * p.RM * (p.RM * FCs + as_ * p.LM) - p.LM * FCs) / (
        (wsc + eps) * (w0 * CEQ * p.LM * p.RM)**2 * FCs
    )
    B = p.Vpk / (w0**2 * CEQ * p.LM * p.RM)
    return _slow_response(t, p, A, B, as_, FCs, wsc, disc)


def slow_zero(t, p):
    CEQ, CT, w0, af, wfc, as_, ws, wsc, FCs, disc = derived(p)
    eps = 1e-10
    # Eq. 19, 20
    A = p.Vpk * (FCs - as_ * CEQ * p.RM) / (
        (wsc + eps) * w0 * CEQ * p.LM * p.RM * FCs
    )
    B = -p.Vpk / (w0 * p.LM)
    return _slow_response(t, p, A, B, as_, FCs, wsc, disc)


def simulate(p, collapse="peak", duration_ms=100):
    t_post = np.linspace(0, duration_ms * 1e-3, 5000)
    w0     = 2 * np.pi * p.f0

    if collapse == "peak":
        Vout = fast_peak(t_post, p) + slow_peak(t_post, p)
    else:
        Vout = fast_zero(t_post, p) + slow_zero(t_post, p)

    t_pre = np.linspace(-1 / p.f0, 0, 1000)
    V_pre = p.Vpk * (np.cos(w0 * t_pre) if collapse == "peak" else np.sin(w0 * t_pre))

    return np.concatenate([t_pre, t_post]), np.concatenate([V_pre, Vout])


def plot_all():
    cvts  = [PROBST, KOJOVIC_1, KOJOVIC_3]
    names = ["Probst", "Kojovic_1", "Kojovic_3"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for row, (p, name) in enumerate(zip(cvts, names)):
        for col, (ctype, title) in enumerate(zip(
            ["peak", "zero"], ["Peak Collapse", "Zero-Crossing Collapse"]
        )):
            t, V = simulate(p, collapse=ctype)
            ax = axes[row][col]
            ax.plot(t * 1e3, V / 1e3, linewidth=1.2)
            ax.axvline(0, color='red', linestyle='--', linewidth=0.8, label='Fault')
            ax.set_title(f"{name} — {title}")
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("Voltage [kV]")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./cvt_transients.png", dpi=150)
    plt.close()
    print("Plot saved.")


def generate_dataset(n_samples=600, noise_std=0.02):
    """
    Labelled waveform windows for ML.
    Label 0 = peak collapse, 1 = zero-crossing collapse.
    """
    cvts    = [PROBST, KOJOVIC_1, KOJOVIC_3]
    X, y    = [], []
    per_cvt = n_samples // (len(cvts) * 2)

    for p in cvts:
        for label, ctype in enumerate(["peak", "zero"]):
            for _ in range(per_cvt):
                dur  = np.random.uniform(80, 120)
                t, V = simulate(p, collapse=ctype, duration_ms=dur)
                V_norm = V / p.Vpk + np.random.normal(0, noise_std, V.shape)

                fault_idx = np.searchsorted(t, 0)
                window    = V_norm[fault_idx: fault_idx + 500]
                if len(window) == 500:
                    X.append(window)
                    y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} timesteps | Labels: {np.bincount(y)}")
    return X, y


if __name__ == "__main__":
    plot_all()
    X, y = generate_dataset()
    np.save("./X_cvt.npy", X)
    np.save("./y_cvt.npy", y)
    print("Dataset saved.")
