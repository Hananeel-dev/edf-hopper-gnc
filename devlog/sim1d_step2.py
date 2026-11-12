"""Step 1: vertical-only hopper sim (altitude, thrust, gravity, drag).
Climb to 10 m, hover, descend, land. No attitude yet."""
import numpy as np
import matplotlib.pyplot as plt

m      = 0.93
g      = 9.81
T_max  = 1.8 * g
tau    = 0.15
CdA    = 1.0 * 0.0123
rho    = 1.2

Kp, Kd = 2.0, 2.5
def target(t):
    if t < 5:     return 2.0 * t, 2.0
    if t < 9:     return 10.0, 0.0
    if t < 9 + 10/1.5: return 10.0 - 1.5 * (t - 9), -1.5
    return 0.0, 0.0

dt, t_end = 0.005, 20.0
h, v, T = 0.0, 0.0, 0.0
log = []
for k in range(int(t_end / dt)):
    t = k * dt
    h_ref, v_ref = target(t)
    a_cmd = Kp * (h_ref - h) + Kd * (v_ref - v) + g
    throttle = np.clip(m * a_cmd / T_max, 0.0, 1.0)
    T += (throttle * T_max - T) * dt / tau
    drag = -0.5 * rho * CdA * v * abs(v)
    a = (T + drag) / m - g
    v += a * dt; h += v * dt
    if h <= 0 and t > 1:
        log.append((t, h_ref, 0, v, throttle)); print(f"touchdown at t={t:.2f}s, speed {v:.2f} m/s"); break
    h = max(h, 0.0); v = max(v, 0.0) if h == 0 else v
    log.append((t, h_ref, h, v, throttle))

t, href, hh, vv, thr = np.array(log).T
fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
ax[0].plot(t, href, "--", label="target"); ax[0].plot(t, hh, label="altitude"); ax[0].set_ylabel("m"); ax[0].legend()
ax[1].plot(t, vv); ax[1].set_ylabel("vertical speed, m/s")
ax[2].plot(t, thr); ax[2].set_ylabel("throttle 0-1"); ax[2].set_xlabel("time, s")
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("step2.png", dpi=100); plt.show()
print(f"end: t={t[-1]:.2f}s  altitude={hh[-1]:.2f} m  speed={vv[-1]:.2f} m/s  max throttle={thr.max():.2f}")
