"""Step 9: priorities. During the braking burn, tilt is capped at 2 deg (soft landing first);
full sideways correction returns in the slow terminal descent.
Step 8 header: 2-D (planar) hoverslam. The vehicle can now tilt (pitch) and drift sideways.
The gimbal tilts the thrust to steer. States: x (sideways), z (height), theta (pitch).
Simplifications for now: attitude sensor = truth + noise; wind = a side gust during the fall."""
import numpy as np
import matplotlib.pyplot as plt

g, rho, CdA, Cd_side_A = 9.81, 1.2, 0.0123, 1.0 * 0.05
m_nom, T_nom = 0.93, 1.8 * g
I_yy = 0.012
L    = 0.12
GMAX, GRATE, GTAU = np.radians(10), np.radians(100), 0.03
idle, h0, plan, tau_plan = 0.10, 30.0, 0.70, 0.25
h_gate, v_gate, Ki, Kv = 1.5, -0.5, 3.0, 3.0
a_burn = plan * T_nom / m_nom - g
WN, ZETA = 8.0, 0.9
Kx, Kvx, TILT_MAX = 0.04, 0.12, np.radians(8)
dt = 0.005

def fly(T_true, tau_true, m_true, gust, theta0, misalign, rng, log=False):
    x, z, vx, vz, th, w = 0.0, h0, 0.0, 0.0, theta0, 0.0
    T, dg, t, phase = m_true * g, 0.0, 0.0, "hover"
    integ, events, rec = 0.0, [], []
    def mark(name): events.append((name, t, x, z))
    mark("start hover")
    while True:
        th_m, w_m = th + rng.normal(0, np.radians(0.3)), w + rng.normal(0, np.radians(1))
        z_e, vz_e = z + rng.normal(0, 0.1), vz + rng.normal(0, 0.05)
        x_e, vx_e = x + rng.normal(0, 0.2), vx + rng.normal(0, 0.1)

        if phase == "hover" and t >= 2.0: phase = "fall"; mark("idle / fall")
        if phase == "fall" and vz_e < 0 and z_e <= (vz_e**2 - v_gate**2)/(2*a_burn) + abs(vz_e)*tau_plan + h_gate:
            phase = "burn"; mark("ignition")
        if phase == "burn" and (z_e <= h_gate or vz_e >= v_gate): phase = "terminal"; mark("gate")
        if phase == "hover":  a_cmd = Kv * (0 - vz_e)
        elif phase == "fall": a_cmd = None
        elif phase == "burn": a_cmd = (vz_e**2 - v_gate**2) / (2*max(z_e - h_gate, 0.1))
        else:                 a_cmd = Kv * (v_gate - vz_e)
        a_meas = (T*np.cos(th + dg) / m_true - g) + rng.normal(0, 0.3)
        if a_cmd is None: throttle, integ = idle, 0.0
        else:
            throttle = m_nom*(g + a_cmd)/(T_nom*max(np.cos(th), 0.8)) + Ki*integ*m_nom/T_nom
            if 0 < throttle < 1: integ += (a_cmd - a_meas) * dt
            throttle = float(np.clip(throttle, 0, 1))

        tilt_cap = np.radians(2) if phase == "burn" else TILT_MAX
        th_ref = np.clip(-(Kx*x_e + Kvx*vx_e), -tilt_cap, tilt_cap)
        if phase == "fall": th_ref = 0.0
        alpha_des = -(WN**2*(th_m - th_ref) + 2*ZETA*WN*w_m)
        T_est = max(throttle * T_nom, 0.5)
        dg_cmd = np.clip(np.arcsin(np.clip(-alpha_des*I_yy/(T_est*L), -1, 1)), -GMAX, GMAX)

        rate = np.clip((dg_cmd - dg)/GTAU, -GRATE, GRATE)
        dg = np.clip(dg + rate*dt, -GMAX, GMAX)

        T += (throttle*T_true - T) * dt / tau_true
        ang = th + dg + misalign
        Fx = T*np.sin(ang) - 0.5*rho*Cd_side_A*vx*abs(vx) + (gust if phase == "fall" else 0.0)
        Fz = T*np.cos(ang) - 0.5*rho*CdA*vz*abs(vz)
        torque = -T*L*np.sin(dg + misalign)
        vx += Fx/m_true*dt; vz += (Fz/m_true - g)*dt; w += torque/I_yy*dt
        x += vx*dt; z += vz*dt; th += w*dt; t += dt
        if log: rec.append((t, x, z, np.degrees(th), np.degrees(dg), throttle))
        if z <= 0:
            mark("touchdown")
            res = (-vz, abs(x), np.degrees(abs(th)), max(abs(r[3]) for r in rec) if log else None)
            return (res, events, rec) if log else (-vz, abs(x), np.degrees(abs(th)))
        if t > 30 or abs(th) > np.radians(60): return (99, abs(x), 99) if not log else ((99,0,99,0), events, rec)

rng = np.random.default_rng(2)
res, ev, rec = fly(T_nom*0.85, 0.25, m_nom*1.05, gust=0.8, theta0=np.radians(3), misalign=np.radians(0.7), rng=rng, log=True)
print(f"detailed run: touchdown {res[0]:.2f} m/s, lands {res[1]:.2f} m from pad, tilt at touchdown {res[2]:.1f} deg, max tilt {res[3]:.1f} deg")
for name, t, x, z in ev: print(f"  {name:12s} t={t:5.2f}s  x={x:+.2f} m  z={z:5.1f} m")

out = []
for _ in range(300):
    out.append(fly(T_nom*rng.uniform(0.80,1.05), rng.uniform(0.10,0.30), m_nom*rng.uniform(0.97,1.08),
                   gust=rng.uniform(-1.0,1.0), theta0=np.radians(rng.uniform(-3,3)), misalign=np.radians(rng.uniform(-1,1)), rng=rng))
v_td, miss, tilt = np.array(out).T
ok = (v_td <= 1) & (tilt <= 5)
print(f"Monte Carlo 300: soft & upright {ok.mean()*100:.0f}%   worst speed {v_td.max():.2f} m/s   worst miss {miss.max():.2f} m   worst tilt {tilt.max():.1f} deg")

t_, x_, z_, th_, dg_, thr_ = map(np.array, zip(*rec))
fig = plt.figure(figsize=(14, 8))
ax0 = fig.add_subplot(1, 2, 1)
ax0.plot(x_, z_, lw=2)
cols = {"start hover":"k", "idle / fall":"tab:orange", "ignition":"r", "gate":"tab:purple", "touchdown":"g"}
for name, t, x, z in ev:
    ax0.plot(x, z, "o", ms=10, color=cols[name]); ax0.annotate(f" {name}\n t={t:.1f}s", (x, z), fontsize=9)
ax0.axhline(0, color="saddlebrown", lw=3); ax0.set_xlabel("sideways, m"); ax0.set_ylabel("height, m")
ax0.set_title("step 9: tilt capped at 2 deg during the burn"); ax0.grid(alpha=0.3)
ax0.set_xlim(min(x_.min(), -1) - 0.5, max(x_.max(), 1) + 1.5)
ax1 = fig.add_subplot(3, 2, 2); ax1.plot(t_, th_, label="pitch (tilt)"); ax1.plot(t_, dg_, label="gimbal"); ax1.set_ylabel("deg"); ax1.legend()
ax2 = fig.add_subplot(3, 2, 4); ax2.plot(t_, z_); ax2.set_ylabel("height, m")
ax3 = fig.add_subplot(3, 2, 6); ax3.plot(t_, thr_); ax3.set_ylabel("throttle"); ax3.set_xlabel("time, s")
for a in (ax1, ax2, ax3):
    a.grid(alpha=0.3)
    for name, t, x, z in ev: a.axvline(t, color=cols[name], ls="--", lw=0.8)
plt.tight_layout(); plt.savefig("step9.png", dpi=100); plt.show()
