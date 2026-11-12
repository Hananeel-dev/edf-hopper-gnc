"""Step 10: full mission, pad to pad, in 2-D (pitch plane) + a roll channel.
New: cradle + liftoff, ground effect, climb, go/no-go, idle cut at the top, leg deploy at ignition,
battery sag, and roll: residual fan swirl, spool-up reaction kicks, vane control, gyro limit."""
import numpy as np
import matplotlib.pyplot as plt

g, rho = 9.81, 1.2
m_nom   = 0.935
T_nom   = 1.8 * g
I_yy, L = 0.012, 0.12
I_roll  = 1.5e-3
I_rotor, W_MAX = 8e-6, 4500.0
CdA_up, CdA_side = 0.0123, 0.05
GMAX, GRATE, GTAU = np.radians(10), np.radians(100), 0.03
VMAX, VRATE, VTAU = np.radians(20), np.radians(600), 0.02
K_VANE  = 0.0095
GYRO_MAX = np.radians(2000)
CAP_mAh, I_MAX = 1300.0, 75.0
H_CUT, CLIMB = 30.0, 3.0
plan, tau_plan, h_gate, v_gate, Ki, Kv = 0.70, 0.25, 1.5, -0.5, 3.0, 3.0
WN, ZETA = 8.0, 0.9
Kx, Kvx, TILT_MAX = 0.04, 0.12, np.radians(8)
dt = 0.005

def fly(rng, T_scale=1.0, tau=0.15, m_scale=1.0, gust=0.0, res_torque=0.008, idle=0.10,
        ign_ramp=0.0, leg_kick=0.0, misalign=0.0, log=False):
    m = m_nom * m_scale
    a_burn = plan * T_nom / m_nom - g
    x = z = vx = vz = th = w = 0.0
    p = 0.0
    T, dg, dv, used = 0.0, 0.0, 0.0, 0.0
    w_rot, t, phase, integ, ramp_t0 = 0.0, 0.0, "pad", 0.0, None
    legs_out, ev, rec, peak_roll, sat_t = False, [], [], 0.0, 0.0
    def mark(n): ev.append((n, t, x, z))
    mark("pad")
    while True:
        V = 1.0 - 0.10 * used / CAP_mAh - 0.04 * (T / T_nom)
        T_avail = T_nom * T_scale * V**2
        th_m = th + rng.normal(0, np.radians(0.3)); w_m = w + rng.normal(0, np.radians(1))
        p_m  = np.clip(p + rng.normal(0, np.radians(1)), -GYRO_MAX, GYRO_MAX)
        z_e, vz_e = z + rng.normal(0, 0.1), vz + rng.normal(0, 0.05)
        x_e, vx_e = x + rng.normal(0, 0.2), vx + rng.normal(0, 0.1)
        if phase == "pad" and t >= 2.0: phase = "spool"; mark("arm / spool-up")
        if phase == "spool" and z > 0.3: phase = "climb"; mark("liftoff")
        if phase == "climb" and z_e >= H_CUT:
            go = abs(th_m) < np.radians(5) and V > 0.85
            phase = "coast" if go else "abort_land"; mark("GO: idle cut" if go else "NO-GO: land")
        if phase == "coast" and vz_e < 0: phase = "fall"; mark("apex")
        if phase == "fall" and z_e <= (vz_e**2 - v_gate**2)/(2*a_burn) + abs(vz_e)*(tau_plan + ign_ramp/2) + h_gate:
            phase, ramp_t0, legs_out = "burn", t, True; mark("ignition + legs")
        if phase == "burn" and (z_e <= h_gate or vz_e >= v_gate): phase = "terminal"; mark("gate")
        if phase == "pad":       a_cmd = None; thr = 0.0
        elif phase == "spool":   a_cmd = None; thr = min(1.0, (t - 2.0) / 1.0) * 0.65
        elif phase == "climb":   a_cmd = Kv * (CLIMB - vz_e)
        elif phase in ("coast", "fall"): a_cmd = None; thr = idle
        elif phase == "burn":    a_cmd = (vz_e**2 - v_gate**2) / (2*max(z_e - h_gate, 0.1))
        elif phase == "abort_land": a_cmd = Kv * (-1.0 - vz_e)
        else:                    a_cmd = Kv * (v_gate - vz_e)
        a_meas = (T*np.cos(th+dg)/m - g if z > 0.01 else 0.0) + rng.normal(0, 0.3)
        if a_cmd is not None:
            thr = m_nom*(g + a_cmd)/(T_nom*max(np.cos(th), 0.8)) + Ki*integ*m_nom/T_nom
            if 0 < thr < 1: integ += (a_cmd - a_meas) * dt
            thr = float(np.clip(thr, 0, 1))
            if phase == "burn" and ign_ramp > 0:
                thr = min(thr, idle + (1 - idle) * (t - ramp_t0) / ign_ramp)
        else: integ = 0.0
        if thr >= 0.999: sat_t += dt
        cap = np.radians(3) if phase in ("coast", "fall") else TILT_MAX
        th_ref = np.clip(-(Kx*x_e + Kvx*vx_e), -cap, cap) if z > 0.3 else 0.0
        T_est = max(thr * T_nom, 0.5)
        alpha = -(WN**2*(th_m - th_ref) + 2*ZETA*WN*w_m)
        dg_cmd = np.clip(np.arcsin(np.clip(-alpha*I_yy/(T_est*L), -1, 1)), -GMAX, GMAX)
        dg = np.clip(dg + np.clip((dg_cmd - dg)/GTAU, -GRATE, GRATE)*dt, -GMAX, GMAX)
        dv_cmd = np.clip(-(6.0 * p_m) * I_roll / (K_VANE * T_est), -VMAX, VMAX)
        dv = np.clip(dv + np.clip((dv_cmd - dv)/VTAU, -VRATE, VRATE)*dt, -VMAX, VMAX)
        T += (thr*T_avail - T) * dt / tau
        w_rot_new = W_MAX * np.sqrt(max(T, 0) / (T_nom * T_scale))
        spool_torque = -I_rotor * (w_rot_new - w_rot) / dt
        w_rot = w_rot_new
        used += I_MAX * (T / (T_nom*T_scale))**1.5 * dt / 3.6
        gnd = 1.0 + 0.15*np.exp(-z/0.15)
        Tg = T * gnd
        ang = th + dg + misalign
        cdz = CdA_up * (1.6 if legs_out else 1.0)
        Fx = Tg*np.sin(ang) - 0.5*rho*CdA_side*vx*abs(vx) + (gust if phase in ("coast","fall") else 0.0)
        Fz = Tg*np.cos(ang) - 0.5*rho*cdz*vz*abs(vz) - m*g
        tq_pitch = -T*L*np.sin(dg + misalign)
        if legs_out and t - ramp_t0 < 0.3: tq_pitch += leg_kick
        tq_roll = res_torque * (T / (T_nom*0.52)) + spool_torque + K_VANE*T*dv
        if z <= 0.0 and phase in ("pad", "spool"):
            Fz = max(Fz, 0.0); Fx = 0.0; tq_pitch = 0.0; tq_roll = 0.0
        vx += Fx/m*dt; vz += Fz/m*dt; w += tq_pitch/I_yy*dt; p += tq_roll/I_roll*dt
        x += vx*dt; z = max(z + vz*dt, 0.0) if phase in ("pad","spool") else z + vz*dt; th += w*dt; t += dt
        peak_roll = max(peak_roll, abs(p)) if z > 0.3 else peak_roll
        if log: rec.append((t, x, z, np.degrees(th), np.degrees(dg), np.degrees(p), np.degrees(dv), thr, V, used))
        if phase not in ("pad","spool") and z <= 0:
            mark("touchdown")
            r = dict(v=-vz, miss=abs(x), tilt=np.degrees(abs(th)), roll=np.degrees(peak_roll), mAh=used, sat=sat_t)
            return (r, ev, rec) if log else r
        if t > 60 or abs(th) > np.radians(60):
            r = dict(v=99, miss=99, tilt=99, roll=np.degrees(peak_roll), mAh=used, sat=sat_t)
            return (r, ev, rec) if log else r

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    r, ev, rec = fly(rng, T_scale=0.9, tau=0.2, m_scale=1.03, gust=0.6, leg_kick=0.02, misalign=np.radians(0.5), log=True)
    print("detailed run (90% thrust, slow fan, gust, leg kick):")
    print(f"  touchdown {r['v']:.2f} m/s, {r['miss']:.2f} m from pad, tilt {r['tilt']:.1f} deg, "
          f"peak roll rate {r['roll']:.0f} deg/s, battery used {r['mAh']:.0f} of 1300 mAh, at 100% throttle {r['sat']:.2f} s")
    for n, t, x, z in ev: print(f"  {n:16s} t={t:5.2f}s  x={x:+.2f}  z={z:5.1f}")

    def mc(idle, ramp, N=150):
        rng = np.random.default_rng(7); out = []
        for _ in range(N):
            out.append(fly(rng, T_scale=rng.uniform(0.85,1.05), tau=rng.uniform(0.1,0.3), m_scale=rng.uniform(0.97,1.06),
                           gust=rng.uniform(-0.8,0.8), res_torque=rng.uniform(-0.015,0.015), idle=idle, ign_ramp=ramp,
                           leg_kick=rng.uniform(-0.03,0.03), misalign=np.radians(rng.uniform(-0.7,0.7))))
        v = np.array([o["v"] for o in out]); roll = np.array([o["roll"] for o in out]); tilt = np.array([o["tilt"] for o in out])
        ok = (v <= 1) & (tilt <= 5)
        return ok.mean()*100, np.median(roll), roll.max(), max(o["mAh"] for o in out)
    print("\nMonte Carlo, 150 runs each:  idle  ramp | soft&upright | roll rate median / worst | worst battery use")
    for idle, ramp in [(0.10, 0.0), (0.10, 0.5), (0.40, 0.0), (0.40, 0.5)]:
        ok, rmed, rmax, mah = mc(idle, ramp)
        print(f"                              {idle:.2f}  {ramp:.1f}s |    {ok:5.0f}%    |   {rmed:5.0f} / {rmax:5.0f} deg/s   | {mah:5.0f} mAh")

    t_, x_, z_, th_, dg_, p_, dv_, thr_, V_, mah_ = map(np.array, zip(*rec))
    cols = {"pad":"k","arm / spool-up":"tab:brown","liftoff":"tab:blue","GO: idle cut":"tab:cyan","NO-GO: land":"m",
            "apex":"tab:orange","ignition + legs":"r","gate":"tab:purple","touchdown":"g"}
    fig = plt.figure(figsize=(15, 9))
    a0 = fig.add_subplot(1, 2, 1); a0.plot(x_, z_, lw=2)
    for n, t, x, z in ev:
        a0.plot(x, z, "o", ms=9, color=cols[n]); a0.annotate(f"  {n} ({t:.1f}s)", (x, z), fontsize=9)
    a0.axhline(0, color="saddlebrown", lw=3); a0.set_xlabel("sideways, m"); a0.set_ylabel("height, m")
    a0.set_title("pad-to-pad mission"); a0.grid(alpha=0.3); a0.set_xlim(min(x_.min(), -1) - 0.5, max(x_.max(), 1) + 3)
    panels = [(th_, "pitch, deg", dg_, "gimbal"), (p_, "roll rate, deg/s", dv_ * 10, "vane x10"),
              (thr_, "throttle", V_, "battery V (fraction)")]
    for i, (y1, l1, y2, l2) in enumerate(panels):
        a = fig.add_subplot(3, 2, 2*i + 2); a.plot(t_, y1, label=l1); a.plot(t_, y2, label=l2, alpha=0.7); a.legend(fontsize=8); a.grid(alpha=0.3)
        for n, t, x, z in ev: a.axvline(t, color=cols[n], ls="--", lw=0.7)
    a.set_xlabel("time, s")
    plt.tight_layout(); plt.savefig("step10.png", dpi=100); plt.show()
