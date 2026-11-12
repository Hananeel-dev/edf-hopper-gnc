"""Step 4: Monte Carlo of the 1-D hoverslam.
The guidance still ASSUMES the nominal vehicle (1.8 kg thrust, 0.15 s lag, perfect altitude).
Reality is drawn at random each run. How often does it still land softly?"""
import numpy as np
import matplotlib.pyplot as plt

g, rho, CdA = 9.81, 1.2, 0.0123
m_nom, T_nom, tau_nom = 0.93, 1.8 * g, 0.15
idle, plan, h0 = 0.10, 0.80, 30.0
a_burn = plan * T_nom / m_nom - g

def fly(T_true, tau_true, m_true, alt_bias, alt_noise, rng, dt=0.005):
    h, v, T, t, phase = h0, 0.0, m_true * g, 0.0, "hover"
    while True:
        h_meas = h + alt_bias + rng.normal(0, alt_noise)
        if phase == "hover" and t >= 1.0: phase = "fall"
        if phase == "fall":
            if v < 0 and h_meas <= v * v / (2 * a_burn) + abs(v) * tau_nom:
                phase = "burn"
        if phase == "hover":   throttle = m_nom * g / T_nom
        elif phase == "fall":  throttle = idle
        else:
            a_req = v * v / (2 * max(h_meas, 0.05))
            throttle = np.clip(m_nom * (g + a_req) / T_nom, 0, 1)
        T += (throttle * T_true - T) * dt / tau_true
        a = (T - 0.5 * rho * CdA * v * abs(v)) / m_true - g
        v += a * dt; h += v * dt; t += dt
        if h <= 0: return -v, 0.0
        if phase == "burn" and v >= 0: return 0.0, h

rng = np.random.default_rng(1)
N = 500
res = []
for _ in range(N):
    T_true   = T_nom * rng.uniform(0.80, 1.05)
    tau_true = rng.uniform(0.10, 0.30)
    m_true   = m_nom * rng.uniform(0.97, 1.08)
    bias     = rng.uniform(-0.5, 0.5)
    res.append(fly(T_true, tau_true, m_true, bias, 0.2, rng) + (T_true / T_nom, tau_true))
res = np.array(res)
v_td, h_stop, Tf, tf = res.T
air = h_stop > 0
soft = (v_td <= 1.0) & ~air
print(f"soft landings (<= 1 m/s): {soft.mean()*100:.0f}%")
print(f"hard (1-3 m/s): {((v_td>1)&(v_td<=3)).mean()*100:.0f}%   crash (> 3 m/s): {(v_td>3).mean()*100:.0f}%")
print(f"stopped in the air (would then drop): {air.mean()*100:.0f}%")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].hist(v_td[~air], bins=40); ax[0].axvline(1, color="g", ls="--"); ax[0].axvline(3, color="r", ls="--")
ax[0].set_xlabel("touchdown speed, m/s"); ax[0].set_ylabel("runs"); ax[0].set_title("500 hoverslams")
sc = ax[1].scatter(Tf[~air] * 100, v_td[~air], c=tf[~air], cmap="viridis", s=12)
ax[1].set_xlabel("real thrust, % of assumed"); ax[1].set_ylabel("touchdown speed, m/s")
plt.colorbar(sc, ax=ax[1], label="real spool lag, s"); ax[1].set_title("what drives a hard landing")
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("step4.png", dpi=100); plt.show()
