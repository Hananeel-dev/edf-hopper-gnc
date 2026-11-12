"""Step 11: full 6-DOF. Pitch, yaw and roll all coupled.
New: quaternion attitude, 2-axis gimbal in the body frame, rotor gyroscopic torque,
servos that only update at 50 Hz, gyro clipping that corrupts the heading estimate,
and a test: does the controller need to know which way the body has rolled?"""
import math, random
import numpy as np
import matplotlib.pyplot as plt

g, rho = 9.81, 1.2
m_nom, T_nom = 0.935, 1.8 * 9.81
I_xx = I_yy = 0.012; I_zz = 1.5e-3
L = 0.12
I_rotor, W_MAX = 8e-6, 4500.0
CdA_up, CdA_side = 0.0123, 0.05
GMAX, GRATE, GTAU = math.radians(10), math.radians(100), 0.03
VMAX, VRATE, VTAU = math.radians(20), math.radians(600), 0.02
K_VANE = 0.0095
GYRO_MAX = math.radians(2000)
CAP_mAh, I_MAX = 1300.0, 75.0
H_CUT, CLIMB = 30.0, 3.0
plan, tau_plan, h_gate, v_gate, Ki, Kv = 0.70, 0.25, 1.5, -0.5, 3.0, 3.0
WN, ZETA = 8.0, 0.9
Kx, Kvx, TILT_MAX = 0.04, 0.12, math.radians(8)
dt, SERVO_DT = 0.004, 0.02

def qmul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return (w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2)
def qnorm(q):
    n = math.sqrt(sum(c*c for c in q)); return tuple(c/n for c in q)
def rot(q, v):
    w,x,y,z = q
    return (v[0]*(1-2*(y*y+z*z)) + v[1]*2*(x*y-w*z) + v[2]*2*(x*z+w*y),
            v[0]*2*(x*y+w*z) + v[1]*(1-2*(x*x+z*z)) + v[2]*2*(y*z-w*x),
            v[0]*2*(x*z-w*y) + v[1]*2*(y*z+w*x) + v[2]*(1-2*(x*x+y*y)))
def rot_T(q, v):
    w,x,y,z = q; return rot((w,-x,-y,-z), v)
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def clip(v, lo, hi): return lo if v < lo else hi if v > hi else v
def servo(cur, cmd, tau, rate, lim, h):
    return clip(cur + clip((cmd - cur)/tau, -rate, rate)*h, -lim, lim)

def fly(seed, T_scale=1.0, tau=0.15, m_scale=1.0, gust=(0.0, 0.0), res_torque=0.008, idle=0.40,
        leg_kick=(0.0, 0.0), misalign=(0.0, 0.0), roll_aware=True, gyroscopic=True, log=False):
    rnd = random.Random(seed); n = rnd.gauss
    m = m_nom*m_scale; a_burn = plan*T_nom/m_nom - g
    r = [0.0, 0.0, 0.0]; v = [0.0, 0.0, 0.0]
    q = (1.0, 0.0, 0.0, 0.0); w = [0.0, 0.0, 0.0]
    T = used = 0.0; ga = gb = dv = 0.0; ga_c = gb_c = dv_c = 0.0
    w_rot = 0.0; t = 0.0; phase = "pad"; integ = 0.0; ign_t = None; legs = False
    head_err = 0.0
    servo_clock = 0.0; ev = []; rec = []; peak_roll = peak_tilt = 0.0
    def mark(s): ev.append((s, t, r[0], r[1], r[2]))
    mark("pad")
    while True:
        V = 1.0 - 0.10*used/CAP_mAh - 0.04*(T/T_nom); T_avail = T_nom*T_scale*V*V
        w_m = [clip(w[i] + n(0, math.radians(1)), -GYRO_MAX, GYRO_MAX) for i in range(3)]
        head_err += (w[2] - w_m[2])*dt
        q_est = qmul(q, (math.cos(-head_err/2), 0, 0, math.sin(-head_err/2)))
        z_e, vz_e = r[2] + n(0, 0.1), v[2] + n(0, 0.05)
        x_e = (r[0] + n(0, 0.2), r[1] + n(0, 0.2)); vx_e = (v[0] + n(0, 0.1), v[1] + n(0, 0.1))
        zb = rot(q, (0, 0, 1)); tilt = math.acos(clip(zb[2], -1, 1))
        if phase == "pad" and t >= 2.0: phase = "spool"; mark("arm / spool-up")
        if phase == "spool" and r[2] > 0.3: phase = "climb"; mark("liftoff")
        if phase == "climb" and z_e >= H_CUT:
            go = tilt < math.radians(5) and V > 0.85
            phase = "coast" if go else "abort"; mark("GO: idle cut" if go else "NO-GO")
        if phase == "coast" and vz_e < 0: phase = "fall"; mark("apex")
        if phase == "fall" and z_e <= (vz_e**2 - v_gate**2)/(2*a_burn) + abs(vz_e)*tau_plan + h_gate:
            phase, ign_t, legs = "burn", t, True; mark("ignition + legs")
        if phase == "burn" and (z_e <= h_gate or vz_e >= v_gate): phase = "terminal"; mark("gate")
        a_cmd = None
        if phase == "pad": thr = 0.0
        elif phase == "spool": thr = min(1.0, (t-2.0)/1.0)*0.65
        elif phase == "climb": a_cmd = Kv*(CLIMB - vz_e)
        elif phase in ("coast", "fall"): thr = idle
        elif phase == "burn": a_cmd = (vz_e**2 - v_gate**2)/(2*max(z_e - h_gate, 0.1))
        elif phase == "abort": a_cmd = Kv*(-1.0 - vz_e)
        else: a_cmd = Kv*(v_gate - vz_e)
        if a_cmd is not None:
            a_meas = (T*zb[2]/m - g if r[2] > 0.01 else 0.0) + n(0, 0.3)
            thr = m_nom*(g + a_cmd)/(T_nom*max(zb[2], 0.8)) + Ki*integ*m_nom/T_nom
            if 0 < thr < 1: integ += (a_cmd - a_meas)*dt
            thr = clip(thr, 0.0, 1.0)
        else: integ = 0.0
        servo_clock += dt
        if servo_clock >= SERVO_DT - 1e-9:
            servo_clock = 0.0
            cap = math.radians(3) if phase in ("coast", "fall") else TILT_MAX
            if r[2] > 0.3:
                tx = clip(-(Kx*x_e[0] + Kvx*vx_e[0]), -cap, cap); ty = clip(-(Kx*x_e[1] + Kvx*vx_e[1]), -cap, cap)
            else: tx = ty = 0.0
            zd = (math.sin(tx), math.sin(ty), 1.0); nz = math.sqrt(sum(c*c for c in zd)); zd = tuple(c/nz for c in zd)
            zb_e = rot(q_est, (0, 0, 1))
            e_w = cross(zb_e, zd)
            e_b = rot_T(q_est, e_w) if roll_aware else e_w
            T_est = max(thr*T_nom, 0.5)
            ax = WN**2*e_b[0] - 2*ZETA*WN*w_m[0]; ay = WN**2*e_b[1] - 2*ZETA*WN*w_m[1]
            ga_c = clip(math.asin(clip(-ax*I_xx/(L*T_est), -1, 1)), -GMAX, GMAX)
            gb_c = clip(math.asin(clip(-ay*I_yy/(L*T_est), -1, 1)), -GMAX, GMAX)
            dv_c = clip(-6.0*w_m[2]*I_zz/(K_VANE*T_est), -VMAX, VMAX)
        ga = servo(ga, ga_c, GTAU, GRATE, GMAX, dt); gb = servo(gb, gb_c, GTAU, GRATE, GMAX, dt)
        dv = servo(dv, dv_c, VTAU, VRATE, VMAX, dt)
        T += (thr*T_avail - T)*dt/tau
        w_new = W_MAX*math.sqrt(max(T, 0)/(T_nom*T_scale)); spool = -I_rotor*(w_new - w_rot)/dt; w_rot = w_new
        used += I_MAX*(T/(T_nom*T_scale))**1.5*dt/3.6
        gnd = 1.0 + 0.15*math.exp(-max(r[2], 0)/0.15); Tg = T*gnd
        a1, b1 = ga + misalign[0], gb + misalign[1]
        Fb = (Tg*math.sin(b1)*math.cos(a1), -Tg*math.sin(a1), Tg*math.cos(a1)*math.cos(b1))
        tau_b = [L*Fb[1], -L*Fb[0], res_torque*(T/(T_nom*0.52)) + spool + K_VANE*T*dv]
        if legs and t - ign_t < 0.3: tau_b[0] += leg_kick[0]; tau_b[1] += leg_kick[1]
        h = (0.0, 0.0, I_rotor*w_rot if gyroscopic else 0.0)
        Iw = (I_xx*w[0] + h[0], I_yy*w[1] + h[1], I_zz*w[2] + h[2])
        gyro_t = cross(w, Iw)
        Fw = list(rot(q, Fb)); cdz = CdA_up*(1.6 if legs else 1.0)
        for i, cd in ((0, CdA_side), (1, CdA_side), (2, cdz)): Fw[i] -= 0.5*rho*cd*v[i]*abs(v[i])
        Fw[2] -= m*g
        if phase in ("coast", "fall"): Fw[0] += gust[0]; Fw[1] += gust[1]
        if r[2] <= 0.0 and phase in ("pad", "spool"):
            Fw = [0.0, 0.0, max(Fw[2], 0.0)]; tau_b = [0, 0, 0]; gyro_t = (0, 0, 0)
        for i, I in enumerate((I_xx, I_yy, I_zz)): w[i] += (tau_b[i] - gyro_t[i])/I*dt
        for i in range(3): v[i] += Fw[i]/m*dt; r[i] += v[i]*dt
        if phase in ("pad", "spool"): r[2] = max(r[2], 0.0)
        q = qnorm(qmul(q, (1.0, 0.5*w[0]*dt, 0.5*w[1]*dt, 0.5*w[2]*dt)))
        t += dt
        if r[2] > 0.3: peak_roll = max(peak_roll, abs(w[2])); peak_tilt = max(peak_tilt, tilt)
        if log: rec.append((t, r[0], r[1], r[2], math.degrees(tilt), math.degrees(ga), math.degrees(gb),
                            math.degrees(w[2]), math.degrees(head_err), thr))
        if phase not in ("pad", "spool") and r[2] <= 0:
            mark("touchdown")
            res = dict(v=-v[2], miss=math.hypot(r[0], r[1]), tilt=math.degrees(tilt), roll=math.degrees(peak_roll),
                       ptilt=math.degrees(peak_tilt), head=math.degrees(abs(head_err)))
            return (res, ev, rec) if log else res
        if t > 60 or tilt > math.radians(60):
            res = dict(v=99, miss=99, tilt=99, roll=math.degrees(peak_roll), ptilt=99, head=math.degrees(abs(head_err)))
            return (res, ev, rec) if log else res

def mc(N=80, **kw):
    R = random.Random(11); out = []
    for k in range(N):
        out.append(fly(k, T_scale=R.uniform(0.85, 1.05), tau=R.uniform(0.1, 0.3), m_scale=R.uniform(0.97, 1.06),
                       gust=(R.uniform(-0.8, 0.8), R.uniform(-0.8, 0.8)), res_torque=R.uniform(-0.015, 0.015),
                       leg_kick=(R.uniform(-0.03, 0.03), R.uniform(-0.03, 0.03)),
                       misalign=(math.radians(R.uniform(-0.7, 0.7)), math.radians(R.uniform(-0.7, 0.7))), **kw))
    v = np.array([o["v"] for o in out]); tl = np.array([o["tilt"] for o in out])
    ok = (v <= 1) & (tl <= 5)
    return dict(ok=ok.mean()*100, miss=np.percentile([o["miss"] for o in out], 95),
                roll=np.median([o["roll"] for o in out]), ptilt=max(o["ptilt"] for o in out),
                head=max(o["head"] for o in out))

if __name__ == "__main__":
    res, ev, rec = fly(3, T_scale=0.9, tau=0.2, m_scale=1.03, gust=(0.6, -0.4), leg_kick=(0.02, -0.015),
                       misalign=(math.radians(0.5), math.radians(-0.4)), log=True)
    print("detailed 6-DOF run (idle 40%, roll-aware controller):")
    print(f"  touchdown {res['v']:.2f} m/s, {res['miss']:.2f} m from pad, tilt {res['tilt']:.1f} deg, "
          f"peak tilt {res['ptilt']:.1f} deg, peak roll rate {res['roll']:.0f} deg/s")
    for s, t, x, y, z in ev: print(f"  {s:16s} t={t:5.2f}s  x={x:+.2f} y={y:+.2f} z={z:5.1f}")
    print("\nMonte Carlo, 80 runs each:")
    print("  case                               | soft&upright | 95% miss | median roll | worst tilt | worst heading err")
    cases = [("roll-aware, idle 40%", dict(idle=0.40)),
             ("roll-aware, idle 10%", dict(idle=0.10)),
             ("NAIVE (ignores roll), idle 40%", dict(idle=0.40, roll_aware=False)),
             ("NAIVE, idle 10%", dict(idle=0.10, roll_aware=False)),
             ("roll-aware, idle 40%, no gyroscopic", dict(idle=0.40, gyroscopic=False))]
    for name, kw in cases:
        s = mc(**kw)
        print(f"  {name:35s}|    {s['ok']:4.0f}%     | {s['miss']:5.2f} m  | {s['roll']:5.0f} deg/s | {s['ptilt']:5.1f} deg  | {s['head']:5.1f} deg")

    t_, x_, y_, z_, tl_, ga_, gb_, p_, he_, thr_ = map(np.array, zip(*rec))
    fig = plt.figure(figsize=(15, 9))
    a0 = fig.add_subplot(1, 2, 1, projection="3d"); a0.plot(x_, y_, z_, lw=2)
    for s, t, x, y, z in ev: a0.scatter(x, y, z, s=40); a0.text(x, y, z, f" {s} ({t:.1f}s)", fontsize=8)
    a0.set_xlabel("x, m"); a0.set_ylabel("y, m"); a0.set_zlabel("height, m"); a0.set_title("6-DOF pad-to-pad")
    for i, series in enumerate([((tl_, "tilt, deg"), (ga_, "gimbal a, deg"), (gb_, "gimbal b, deg")),
                                ((p_, "roll rate, deg/s"),), ((thr_, "throttle"),)]):
        a = fig.add_subplot(3, 2, 2*i + 2)
        for y, lab in series: a.plot(t_, y, label=lab)
        for s, t, *_ in ev: a.axvline(t, ls="--", lw=0.6, color="grey")
        a.legend(fontsize=8); a.grid(alpha=0.3)
    a.set_xlabel("time, s"); plt.tight_layout(); plt.savefig("step11.png", dpi=100); plt.show()
