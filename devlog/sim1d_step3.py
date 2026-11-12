"""Step 3: 1-D hoverslam. Start hovering at 30 m, drop to idle and fall,
then ONE deceleration burn that should reach ~0 m/s at the ground."""
import numpy as np
import matplotlib.pyplot as plt

m, g   = 0.93, 9.81
T_max  = 1.8 * g
tau    = 0.15
CdA, rho = 1.0 * 0.0123, 1.2
idle   = 0.10
plan   = 0.80
h0     = 30.0

a_burn = plan * T_max / m - g
dt = 0.005
h, v, T = h0, 0.0, m * g
phase, log, t = "hover", [], 0.0
while True:
    if phase == "hover" and t >= 1.0:
        phase = "fall"
    if phase == "fall":
        stop_dist = v * v / (2 * a_burn) + abs(v) * tau
        if v < 0 and h <= stop_dist:
            phase = "burn"; t_ign, h_ign, v_ign = t, h, v
    if phase == "hover":
        throttle = m * g / T_max
    elif phase == "fall":
        throttle = idle
    else:
        a_req = v * v / (2 * max(h, 0.05))
        throttle = np.clip(m * (g + a_req) / T_max, 0, 1)
    T += (throttle * T_max - T) * dt / tau
    a = (T - 0.5 * rho * CdA * v * abs(v)) / m - g
    v += a * dt; h += v * dt; t += dt
    log.append((t, h, v, throttle, phase))
    if h <= 0:
        print(f"ignition at {h_ign:.1f} m, {v_ign:.1f} m/s (t={t_ign:.2f}s)")
        print(f"touchdown at t={t:.2f}s, speed {v:.2f} m/s, peak throttle {max(r[3] for r in log):.2f}")
        break
    if phase == "burn" and v >= 0:
        print(f"stopped {h:.2f} m ABOVE ground at t={t:.2f}s (burn too early)"); break

t_, h_, v_, thr_, ph = zip(*log)
fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
ax[0].plot(t_, h_); ax[0].set_ylabel("altitude, m")
ax[1].plot(t_, v_); ax[1].set_ylabel("vertical speed, m/s")
ax[2].plot(t_, thr_); ax[2].set_ylabel("throttle 0-1"); ax[2].set_xlabel("time, s")
for a in ax:
    a.axvline(t_ign, color="r", ls="--", lw=1); a.grid(alpha=0.3)
ax[0].text(t_ign, h0 * 0.9, "  ignition", color="r")
plt.tight_layout(); plt.savefig("step3.png", dpi=100); plt.show()
