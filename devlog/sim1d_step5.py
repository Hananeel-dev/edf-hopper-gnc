"""Step 5: the three fixes, then the same 500-run Monte Carlo.
 A. Accelerometer feedback: compare the deceleration we asked for with what we measure,
    and integrate the difference into the throttle (fixes thrust/mass/lag errors in flight).
 B. Aim for a 'gate' (1.5 m, -0.5 m/s), not zero at zero; then a slow terminal descent
    at hover-like throttle (removes the h -> 0 singularity).
 C. More margin: plan the burn with 70% of assumed thrust, and assume a slow (0.25 s) spool-up."""
import numpy as np
import matplotlib.pyplot as plt

g, rho, CdA = 9.81, 1.2, 0.0123
m_nom, T_nom = 0.93, 1.8 * g
idle, h0 = 0.10, 30.0
plan, tau_plan = 0.70, 0.25
h_gate, v_gate = 1.5, -0.5
Ki = 3.0
Kv = 3.0
a_burn = plan * T_nom / m_nom - g

def fly(T_true, tau_true, m_true, alt_bias, rng, dt=0.005, log=False):
    h, v, T, t, phase = h0, 0.0, m_true * g, 0.0, "hover"
    integ, a_filt, rec = 0.0, 0.0, []
    while True:
        h_meas = h + alt_bias + rng.normal(0, 0.2)
        if phase == "hover" and t >= 1.0: phase = "fall"
        if phase == "fall" and v < 0:
            dist = (v*v - v_gate**2) / (2 * a_burn) + abs(v) * tau_plan + h_gate
            if h_meas <= dist: phase = "burn"
        if phase == "burn" and (h_meas <= h_gate or v >= v_gate): phase = "terminal"

        if phase == "hover":  a_cmd = 0.0
        elif phase == "fall": a_cmd = None
        elif phase == "burn": a_cmd = (v*v - v_gate**2) / (2 * max(h_meas - h_gate, 0.1))
        else:                 a_cmd = Kv * (v_gate - v)

        if a_cmd is None:
            throttle = idle; integ = 0.0
        else:
            err = a_cmd - a_filt
            throttle_ff = m_nom * (g + a_cmd) / T_nom
            throttle = throttle_ff + Ki * integ * m_nom / T_nom
            if 0.0 < throttle < 1.0: integ += err * dt
            throttle = float(np.clip(throttle, 0.0, 1.0))

        T += (throttle * T_true - T) * dt / tau_true
        a = (T - 0.5 * rho * CdA * v * abs(v)) / m_true - g
        a_meas = a + rng.normal(0, 0.3)
        a_filt += (a_meas - a_filt) * dt / 0.03
        v += a * dt; h += v * dt; t += dt
        if log: rec.append((t, h, v, throttle, phase))
        if h <= 0: return (-v, rec) if log else -v
        if t > 30:  return (0.0, rec) if log else 0.0

rng = np.random.default_rng(1)
N, out = 500, []
for _ in range(N):
    Tt, tt, mt, b = T_nom*rng.uniform(0.80,1.05), rng.uniform(0.10,0.30), m_nom*rng.uniform(0.97,1.08), rng.uniform(-0.5,0.5)
    out.append((fly(Tt, tt, mt, b, rng), Tt/T_nom, tt))
v_td, Tf, tf = np.array(out).T
print(f"soft landings (<= 1 m/s): {(v_td<=1).mean()*100:.0f}%")
print(f"hard (1-3 m/s): {((v_td>1)&(v_td<=3)).mean()*100:.0f}%   crash (> 3 m/s): {(v_td>3).mean()*100:.0f}%")
print(f"worst touchdown: {v_td.max():.2f} m/s")

_, rec = fly(T_nom*0.85, 0.25, m_nom*1.05, -0.4, rng, log=True)
t_, h_, v_, thr_, ph_ = zip(*rec)
fig, ax = plt.subplots(2, 2, figsize=(13, 7))
ax[0,0].hist(v_td, bins=40); ax[0,0].axvline(1, color="g", ls="--"); ax[0,0].set_title("500 hoverslams, with fixes"); ax[0,0].set_xlabel("touchdown speed, m/s")
sc = ax[0,1].scatter(Tf*100, v_td, c=tf, s=12); ax[0,1].set_xlabel("real thrust, % of assumed"); ax[0,1].set_ylabel("touchdown speed, m/s"); plt.colorbar(sc, ax=ax[0,1], label="spool lag, s")
ax[1,0].plot(t_, h_); ax[1,0].set_ylabel("altitude, m"); ax[1,0].set_title("one bad case: 85% thrust, +5% mass, 0.25 s lag")
ax[1,1].plot(t_, v_, label="speed"); ax[1,1].plot(t_, np.array(thr_)*10, label="throttle x10"); ax[1,1].legend(); ax[1,1].set_xlabel("time, s")
for a in ax.flat: a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("step5.png", dpi=100); plt.show()
