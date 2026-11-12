"""Step 7: (1) zero the barometer on the pad before takeoff; (2) the filter also estimates the
barometer offset as a 4th state, so when the ToF wakes up it fixes the OFFSET, not the height.
Step 6 header:  no more cheating. The guidance only sees a Kalman-filter estimate of
height and vertical speed, built from noisy accelerometer + barometer + ToF readings.
State x = [h, v, b]  (height, vertical speed, accelerometer bias)."""
import numpy as np
import matplotlib.pyplot as plt

g, rho, CdA = 9.81, 1.2, 0.0123
m_nom, T_nom = 0.93, 1.8 * g
idle, h0, plan, tau_plan = 0.10, 30.0, 0.70, 0.25
h_gate, v_gate, Ki, Kv = 1.5, -0.5, 3.0, 3.0
a_burn = plan * T_nom / m_nom - g
dt = 0.005

ACC_N  = 0.3
BARO_N = 0.3
TOF_N  = 0.02
BARO_EVERY, TOF_EVERY = 4, 4

class Kalman:
    def __init__(self, h_init):
        self.x = np.array([h_init, 0.0, 0.0, 0.0])
        self.P = np.diag([0.1, 0.1, 0.1, 0.15**2])
        self.F = np.array([[1, dt, -0.5*dt*dt, 0], [0, 1, -dt, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = np.array([0.5*dt*dt, dt, 0, 0])
        self.Q = np.diag([1e-6, (ACC_N*dt)**2 * 4, 1e-7, 1e-8])
    def predict(self, a_meas):
        self.x = self.F @ self.x + self.B * a_meas
        self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z, R, H):
        H = np.array(H, dtype=float)
        S = H @ self.P @ H + R
        K = self.P @ H / S
        self.x = self.x + K * (z - H @ self.x)
        self.P = self.P - np.outer(K, H @ self.P)

def fly(T_true, tau_true, m_true, baro_bias, acc_bias, rng, log=False):
    h, v, T, t, phase, k = h0, 0.0, m_true * g, 0.0, "hover", 0
    kf = Kalman(h0)
    integ, a_filt, rec = 0.0, 0.0, []
    while True:
        a_true = (T - 0.5*rho*CdA*v*abs(v)) / m_true - g
        a_meas = a_true + acc_bias + rng.normal(0, ACC_N)
        kf.predict(a_meas)
        if k % BARO_EVERY == 0: kf.update(h + baro_bias + rng.normal(0, BARO_N), BARO_N**2, [1, 0, 0, 1])
        if k % TOF_EVERY == 0 and h < 4.0: kf.update(h + rng.normal(0, TOF_N), TOF_N**2, [1, 0, 0, 0])
        h_e, v_e, b_e, _ = kf.x
        a_filt += ((a_meas - b_e) - a_filt) * dt / 0.03

        if phase == "hover" and t >= 1.0: phase = "fall"
        if phase == "fall" and v_e < 0:
            if h_e <= (v_e*v_e - v_gate**2)/(2*a_burn) + abs(v_e)*tau_plan + h_gate: phase = "burn"
        if phase == "burn" and (h_e <= h_gate or v_e >= v_gate): phase = "terminal"
        if phase == "hover":  a_cmd = Kv * (0 - v_e)
        elif phase == "fall": a_cmd = None
        elif phase == "burn": a_cmd = (v_e*v_e - v_gate**2) / (2*max(h_e - h_gate, 0.1))
        else:                 a_cmd = Kv * (v_gate - v_e)
        if a_cmd is None: throttle, integ = idle, 0.0
        else:
            throttle = m_nom*(g + a_cmd)/T_nom + Ki*integ*m_nom/T_nom
            if 0 < throttle < 1: integ += (a_cmd - a_filt) * dt
            throttle = float(np.clip(throttle, 0, 1))

        T += (throttle*T_true - T) * dt / tau_true
        v += a_true * dt; h += v * dt; t += dt; k += 1
        if log: rec.append((t, h, h_e, v, v_e, throttle))
        if h <= 0: return (-v, rec) if log else -v
        if t > 30: return (0.0, rec) if log else 0.0

rng = np.random.default_rng(1)
out = []
for _ in range(500):
    Tt, tt, mt = T_nom*rng.uniform(0.80,1.05), rng.uniform(0.10,0.30), m_nom*rng.uniform(0.97,1.08)
    bb, ab = rng.uniform(-0.15,0.15), rng.uniform(-0.2,0.2)
    out.append(fly(Tt, tt, mt, bb, ab, rng))
v_td = np.array(out)
print(f"soft (<= 1 m/s): {(v_td<=1).mean()*100:.0f}%   hard (1-3): {((v_td>1)&(v_td<=3)).mean()*100:.0f}%   crash (>3): {(v_td>3).mean()*100:.0f}%")
print(f"worst touchdown: {v_td.max():.2f} m/s")

_, rec = fly(T_nom*0.85, 0.25, m_nom*1.05, 0.15, 0.15, rng, log=True)
t_, h_, he_, v_, ve_, thr_ = map(np.array, zip(*rec))
fig, ax = plt.subplots(2, 2, figsize=(13, 7))
ax[0,0].hist(v_td, bins=40); ax[0,0].axvline(1, color="g", ls="--"); ax[0,0].set_title("500 hoverslams, guidance on ESTIMATES"); ax[0,0].set_xlabel("touchdown speed, m/s")
ax[0,1].plot(t_, h_, label="true height"); ax[0,1].plot(t_, he_, "--", label="Kalman estimate"); ax[0,1].legend(); ax[0,1].set_title("bad case: height")
ax[1,0].plot(t_, v_, label="true speed"); ax[1,0].plot(t_, ve_, "--", label="estimate"); ax[1,0].legend(); ax[1,0].set_xlabel("time, s")
ax[1,1].plot(t_, he_ - h_, label="height error, m"); ax[1,1].plot(t_, ve_ - v_, label="speed error, m/s"); ax[1,1].axhline(0, color="k", lw=0.5); ax[1,1].legend(); ax[1,1].set_xlabel("time, s"); ax[1,1].set_title("estimation error")
for a in ax.flat: a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("step7.png", dpi=100); plt.show()
