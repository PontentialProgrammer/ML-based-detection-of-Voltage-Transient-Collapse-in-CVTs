# ML-based-detection-of-Voltage-Transient-Collapse-in-CVTs

A research project investigating whether machine learning can detect voltage collapse transients in Capacitive Voltage Transformers (CVTs) faster than conventional threshold-based relay methods.

---

## Background

Protection relays in transmission networks monitor system voltage through CVTs. When a fault occurs, CVTs do not immediately reflect the collapse on their secondary side — they lag due to their internal capacitors. This delay impairs protection performance and is a known problem in power systems engineering.

Alonso et al. (2026) derived a closed-form analytical model for predicting CVT transient response during voltage collapse, validated against established EMT simulations. This project uses that analytical model as a data source to generate synthetic CVT waveforms and train a classifier to detect collapse onset as early as possible after a fault.

The core research question is: **given only the secondary voltage signal, how quickly after fault onset can a model reliably detect that a voltage collapse has occurred?**

---

## Approach

1. Implement the analytical model from Alonso et al. to simulate CVT transient waveforms for three transformers (Probst, Kojovic_1, Kojovic_3) under peak and zero-crossing collapse conditions
2. Generate a labelled dataset with Gaussian noise to simulate measurement uncertainty
3. Train a 1D-CNN classifier on progressively shorter post-fault windows to characterise the detection latency vs accuracy tradeoff
4. Compare against a baseline LSTM to evaluate sequential modelling benefit

---

## Repository Structure

```
cvt_simulator.py    analytical model implementation (Alonso et al. Eq. 10-20)
X_cvt.npy           waveform dataset (600 samples x 500 timesteps)
y_cvt.npy           labels (0 = peak collapse, 1 = zero-crossing collapse)
cvt_transients.png  simulated waveforms for all three CVTs
```

---

## CVT Parameters

Parameters are taken directly from Table I of the reference paper:

| CVT | C1 (pF) | C2 (pF) | LC (H) | LM (H) | f0 (Hz) |
|---|---|---|---|---|---|
| Probst | 12730 | 83780 | 69.2 | 6269.7 | 60 |
| Kojovic_1 | 5650 | 81100 | 58.3 | 1000.0 | 60 |
| Kojovic_3 | 6750 | 84110 | 74.4 | 1600.0 | 60 |

---

## Getting Started

```bash
pip install numpy scipy matplotlib
python cvt_simulator.py
```

This generates the waveform plots and saves `X_cvt.npy` and `y_cvt.npy` to disk.

---

## Dependencies

- Python 3.8+
- numpy
- scipy
- matplotlib

---

## Reference

C. H. Alonso, U. Zatika, F. de León, "Analytical Model of Capacitive Voltage Transformers for the Calculation of Voltage Collapse Transients," IEEE Transactions on Power Delivery, 2026. DOI: 10.1109/TPWRD.2026.3665593

---

## Status

Active — simulation and dataset generation complete. Classifier training in progress.
