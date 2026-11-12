"""Step 13: no-GPS reality check, built on step 12.
Compares what the controller knows about sideways position/velocity:
nav = cheat | none | deadreckon | flow | gps | gps+flow."""
import math, random
import numpy as np
import matplotlib.pyplot as plt

g, rho = 9.81, 1.2
m_nom, T_nom = 0.935, 1.8*9.81
I_xx = I_yy = 0.012; I_zz = 1.5e-3; L = 0.12
I_rotor, W_MAX = 8e-6, 4500.0
CdA_up, CdA_side = 0.0123, 0.05
GMAX, GRATE, GTAU = math.radians(10), math.radians(100), 0.03
VMAX, VRATE, VTAU = math.radians(20), math.radians(600), 0.02
K_VANE = 0.0095; GYRO_MAX = math.radians(2000)
CAP_mAh, I_MAX = 1300.0, 75.0
H_CUT, CLIMB = 30.0, 3.0
plan, tau_plan, h_gate, v_gate, Ki, Kv = 0.70, 0.25, 1.5, -0.5, 3.0, 3.0
WN, ZETA = 8.0, 0.9
WN_R, ZETA_R, KI_R = 5.0, 0.9, 8.0
Kx, Kvx, TILT_MAX = 0.04, 0.12, math.radians(8)
dt, SERVO_DT = 0.004, 0.02
PAD_T = 2.0

def qmul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return (w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2)
def qnorm(q): n = math.sqrt(sum(c*c for c in q)); return tuple(c/n for c in q)
def rot(q, v):
    w,x,y,z = q
    return (v[0]*(1-2*(y*y+z*z)) + v[1]*2*(x*y-w*z) + v[2]*2*(x*z+w*y),
            v[0]*2*(x*y+w*z) + v[1]*(1-2*(x*x+z*z)) + v[2]*2*(y*z-w*x),
            v[0]*2*(x*z-w*y) + v[1]*2*(y*z+w*x) + v[2]*(1-2*(x*x+y*y)))
def rot_T(q, v): w,x,y,z = q; return rot((w,-x,-y,-z), v)
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def clip(v, lo, hi): return lo if v < lo else hi if v > hi else v
def servo(cur, cmd, tau, rate, lim, h): return clip(cur + clip((cmd-cur)/tau, -rate, rate)*h, -lim, lim)
def heading(q):
    w,x,y,z = q; return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
def tilt_between(qa, qb):
    a, b = rot(qa, (0,0,1)), rot(qb, (0,0,1)); return math.acos(clip(sum(i*j for i,j in zip(a,b)), -1, 1))

def fly(seed, T_scale=1.0, tau=0.15, m_scale=1.0, gust=(0,0), res_torque=0.008, idle=0.40,
        leg_kick=(0,0), misalign=(0,0), pad_tilt=(0,0), gyro_bias=(0,0,0),
        est="pad_gated", calibrate=True, roll_mode="hold", nav="cheat", term_null=False, log=False):
    """est: 'truth' (step 11 style), 'mahony_always' (trust accel all flight), 'pad_gated' (accel only on the pad)
       roll_mode: 'damp' (step 11) or 'hold' (PID on roll angle)"""
    rnd = random.Random(seed); n = rnd.gauss
    m = m_nom*m_scale; a_burn = plan*T_nom/m_nom - g
    r = [0.0, 0.0, 0.0]; v = [0.0, 0.0, 0.0]
    q = qnorm((1.0, pad_tilt[0]/2, pad_tilt[1]/2, 0.0)); w = [0.0, 0.0, 0.0]
    q_est = (1.0, 0.0, 0.0, 0.0); b_est = [0.0, 0.0, 0.0]; b_sum = [0.0, 0.0, 0.0]; b_n = 0
    bias = list(gyro_bias)
    T = used = 0.0; ga = gb = dv = 0.0; ga_c = gb_c = dv_c = 0.0
    w_rot = 0.0; t = 0.0; phase = "pad"; integ = 0.0; ign_t = None; legs = False
    roll_ref = 0.0; roll_int = 0.0; sf_w = (0.0, 0.0, g)
    servo_clock = 0.0; ev = []; rec = []; peak_err = 0.0; climb_rates = []
    v_dr = [0.0, 0.0]; x_dr = [0.0, 0.0]
    gb_x = [n(0, 1.5), n(0, 1.5)]; gps_clock = 0.0; gps_p = (0.0, 0.0); gps_v = (0.0, 0.0)
    max_range = 0.0; last_t = (0.0, 0.0)
    def mark(s): ev.append((s, t, r[0], r[1], r[2]))
    mark("pad")
    while True:
        V = 1.0 - 0.10*used/CAP_mAh - 0.04*(T/T_nom); T_avail = T_nom*T_scale*V*V
        for i in range(3): bias[i] += n(0, math.radians(0.02))*math.sqrt(dt)
        gyro = [clip(w[i] + bias[i] + n(0, math.radians(0.8)), -GYRO_MAX, GYRO_MAX) for i in range(3)]
        vib = 0.4*T/T_nom*g
        acc = [c + n(0, 0.05*g + vib) for c in rot_T(q, sf_w)]
        on_pad = phase in ("pad",)
        if est == "truth":
            q_est = q; w_hat = [gyro[i] - bias[i] for i in range(3)]
        else:
            if on_pad and calibrate and t > 0.5:
                for i in range(3): b_sum[i] += gyro[i]
                b_n += 1; b_est = [s/b_n for s in b_sum]
            w_hat = [gyro[i] - b_est[i] for i in range(3)]
            use_acc = on_pad or est == "mahony_always"
            if use_acc:
                an = math.sqrt(sum(c*c for c in acc))
                if an > 0.1:
                    a_u = [c/an for c in acc]; g_pred = rot_T(q_est, (0, 0, 1))
                    e = cross(a_u, g_pred)
                    kp = 5.0 if on_pad else 1.0
                    w_hat = [w_hat[i] + kp*e[i] for i in range(3)]
            q_est = qnorm(qmul(q_est, (1.0, 0.5*w_hat[0]*dt, 0.5*w_hat[1]*dt, 0.5*w_hat[2]*dt)))
        err = tilt_between(q, q_est)
        if r[2] > 0.3: peak_err = max(peak_err, err)
        z_e, vz_e = r[2] + n(0, 0.1), v[2] + n(0, 0.05)
        nav_ok_p = nav_ok_v = True
        if nav == "cheat":
            x_e = (r[0]+n(0,0.2), r[1]+n(0,0.2)); vx_e = (v[0]+n(0,0.1), v[1]+n(0,0.1))
        elif nav == "none":
            x_e = vx_e = (0.0, 0.0); nav_ok_p = nav_ok_v = False
        elif nav == "deadreckon":
            if phase != "pad":
                a_w = rot(q_est, acc)
                for i in range(2): v_dr[i] += a_w[i]*dt; x_dr[i] += v_dr[i]*dt
            x_e, vx_e = tuple(x_dr), tuple(v_dr)
        elif nav == "flow":
            h_tof = r[2] + 0.05
            if 0.08 < h_tof < 4.0:
                sig = 0.05 + 0.03*h_tof
                vx_e = (v[0] + n(0, sig), v[1] + n(0, sig))
            else:
                vx_e = (0.0, 0.0); nav_ok_v = False
            x_e = (0.0, 0.0); nav_ok_p = False
        elif nav in ("gps", "gps+flow"):
            for i in range(2): gb_x[i] += (-gb_x[i]/60.0)*dt + n(0, 1.5*math.sqrt(2*dt/60.0))
            gps_clock += dt
            if gps_clock >= 0.1:
                gps_clock = 0.0
                gps_p = (r[0] + gb_x[0] + n(0, 0.3), r[1] + gb_x[1] + n(0, 0.3))
                gps_v = (v[0] + n(0, 0.1), v[1] + n(0, 0.1))
            x_e, vx_e = gps_p, gps_v
            if nav == "gps+flow" and 0.08 < r[2] + 0.05 < 4.0:
                sig = 0.05 + 0.03*(r[2] + 0.05); vx_e = (v[0] + n(0, sig), v[1] + n(0, sig))
        max_range = max(max_range, math.hypot(r[0], r[1]))
        zb = rot(q, (0,0,1)); tilt = math.acos(clip(zb[2], -1, 1))
        zb_est = rot(q_est, (0,0,1)); tilt_est = math.acos(clip(zb_est[2], -1, 1))
        if phase == "pad" and t >= PAD_T: phase = "spool"; mark("arm / spool-up"); roll_ref = heading(q_est)
        if phase == "spool" and r[2] > 0.3: phase = "climb"; mark("liftoff")
        if phase == "climb" and z_e >= H_CUT:
            go = tilt_est < math.radians(5) and V > 0.85
            phase = "coast" if go else "abort"; mark("GO: idle cut" if go else "NO-GO")
        if phase == "coast" and vz_e < 0: phase = "fall"; mark("apex")
        if phase == "fall" and z_e <= (vz_e**2 - v_gate**2)/(2*a_burn) + abs(vz_e)*tau_plan + h_gate:
            phase, ign_t, legs = "burn", t, True; mark("ignition + legs")
        if phase == "burn" and (z_e <= h_gate or vz_e >= v_gate): phase = "terminal"; mark("gate")
        a_cmd = None
        if phase == "pad": thr = 0.0
        elif phase == "spool": thr = min(1.0, (t-PAD_T)/1.0)*0.65
        elif phase == "climb": a_cmd = Kv*(CLIMB - vz_e)
        elif phase in ("coast", "fall"): thr = idle
        elif phase == "burn": a_cmd = (vz_e**2 - v_gate**2)/(2*max(z_e - h_gate, 0.1))
        elif phase == "abort": a_cmd = Kv*(-1.0 - vz_e)
        else: a_cmd = Kv*(v_gate - vz_e)
        if a_cmd is not None:
            a_meas = (T*zb[2]/m - g if r[2] > 0.01 else 0.0) + n(0, 0.3)
            thr = m_nom*(g + a_cmd)/(T_nom*max(zb_est[2], 0.8)) + Ki*integ*m_nom/T_nom
            if 0 < thr < 1: integ += (a_cmd - a_meas)*dt
            thr = clip(thr, 0.0, 1.0)
        else: integ = 0.0
        servo_clock += dt
        if servo_clock >= SERVO_DT - 1e-9:
            servo_clock = 0.0
            T_est = max(thr*T_nom, 0.5)
            if r[2] < 0.05 and phase in ("pad", "spool"):
                ga_c = gb_c = dv_c = 0.0; roll_int = 0.0
            else:
                cap = math.radians(3) if phase in ("coast","fall") else TILT_MAX
                kx = Kx if nav_ok_p else 0.0; kv = Kvx if nav_ok_v else 0.0
                if nav == "flow": kv = 0.25
                if term_null and phase == "terminal" and nav_ok_v:
                    kx, kv = 0.0, 0.25
                tx = clip(-(kx*x_e[0] + kv*vx_e[0]), -cap, cap) if r[2] > 0.3 else 0.0
                ty = clip(-(kx*x_e[1] + kv*vx_e[1]), -cap, cap) if r[2] > 0.3 else 0.0
                if phase == "terminal":
                    if r[2] > 0.3 and (nav_ok_v or nav_ok_p): last_t = (tx, ty)
                    else: tx, ty = last_t
                zd = (math.sin(tx), math.sin(ty), 1.0); nz = math.sqrt(sum(c*c for c in zd)); zd = tuple(c/nz for c in zd)
                e_b = rot_T(q_est, cross(zb_est, zd))
                ax = WN**2*e_b[0] - 2*ZETA*WN*w_hat[0]; ay = WN**2*e_b[1] - 2*ZETA*WN*w_hat[1]
                ga_c = clip(math.asin(clip(-ax*I_xx/(L*T_est), -1, 1)), -GMAX, GMAX)
                gb_c = clip(math.asin(clip(-ay*I_yy/(L*T_est), -1, 1)), -GMAX, GMAX)
                if roll_mode == "damp":
                    az = -6.0*w_hat[2]
                else:
                    e_r = (heading(q_est) - roll_ref + math.pi) % (2*math.pi) - math.pi
                    if thr > 0.3: roll_int = clip(roll_int + e_r*SERVO_DT, -0.5, 0.5)
                    az = -(WN_R**2*e_r + 2*ZETA_R*WN_R*w_hat[2] + KI_R*roll_int)
                dv_c = clip(az*I_zz/(K_VANE*T_est), -VMAX, VMAX)
        ga = servo(ga, ga_c, GTAU, GRATE, GMAX, dt); gb = servo(gb, gb_c, GTAU, GRATE, GMAX, dt)
        dv = servo(dv, dv_c, VTAU, VRATE, VMAX, dt)
        T += (thr*T_avail - T)*dt/tau
        w_new = W_MAX*math.sqrt(max(T,0)/(T_nom*T_scale)); spool = -I_rotor*(w_new - w_rot)/dt; w_rot = w_new
        used += I_MAX*(T/(T_nom*T_scale))**1.5*dt/3.6
        Tg = T*(1.0 + 0.15*math.exp(-max(r[2],0)/0.15))
        a1, b1 = ga + misalign[0], gb + misalign[1]
        Fb = (Tg*math.sin(b1)*math.cos(a1), -Tg*math.sin(a1), Tg*math.cos(a1)*math.cos(b1))
        tau_b = [L*Fb[1], -L*Fb[0], res_torque*(T/(T_nom*0.52)) + spool + K_VANE*T*dv]
        if legs and t - ign_t < 0.3: tau_b[0] += leg_kick[0]; tau_b[1] += leg_kick[1]
        Iw = (I_xx*w[0], I_yy*w[1], I_zz*w[2] + I_rotor*w_rot); gyro_t = cross(w, Iw)
        Fw = list(rot(q, Fb)); cdz = CdA_up*(1.6 if legs else 1.0)
        for i, cd in ((0, CdA_side), (1, CdA_side), (2, cdz)): Fw[i] -= 0.5*rho*cd*v[i]*abs(v[i])
        if phase in ("coast","fall"): Fw[0] += gust[0]; Fw[1] += gust[1]
        in_cradle = r[2] <= 0.0 and phase in ("pad","spool") and Fw[2] < m*g
        if in_cradle:
            sf_w = (0.0, 0.0, g); Fw = [0.0, 0.0, 0.0]; tau_b = [0,0,0]; gyro_t = (0,0,0); w = [0.0,0.0,0.0]
        else:
            sf_w = (Fw[0]/m, Fw[1]/m, Fw[2]/m)
            Fw[2] -= m*g
        for i, I in enumerate((I_xx, I_yy, I_zz)): w[i] += (tau_b[i] - gyro_t[i])/I*dt
        for i in range(3): v[i] += Fw[i]/m*dt; r[i] += v[i]*dt
        if phase in ("pad","spool"): r[2] = max(r[2], 0.0)
        q = qnorm(qmul(q, (1.0, 0.5*w[0]*dt, 0.5*w[1]*dt, 0.5*w[2]*dt))); t += dt
        if phase == "climb" and t > ev[-1][1] + 2: climb_rates.append(abs(w[2]))
        if log: rec.append((t, r[0], r[1], r[2], math.degrees(tilt), math.degrees(tilt_est), math.degrees(err),
                            math.degrees(w[2]), math.degrees(heading(q)), thr))
        done = phase not in ("pad","spool") and r[2] <= 0
        if done or t > 60 or tilt > math.radians(60):
            mark("touchdown" if done else "LOST")
            res = dict(v=-v[2] if done else 99, miss=math.hypot(r[0], r[1]) if done else 99,
                       vh=math.hypot(v[0], v[1]) if done else 99, range=max_range,
                       tilt=math.degrees(tilt) if done else 99, err=math.degrees(peak_err),
                       spin=math.degrees(np.mean(climb_rates)) if climb_rates else 0)
            return (res, ev, rec) if log else res

def mc(N=80, **kw):
    R = random.Random(11); out = []
    for k in range(N):
        out.append(fly(k, T_scale=R.uniform(0.85,1.05), tau=R.uniform(0.1,0.3), m_scale=R.uniform(0.97,1.06),
                       gust=(R.uniform(-0.8,0.8), R.uniform(-0.8,0.8)), res_torque=R.uniform(-0.015,0.015),
                       leg_kick=(R.uniform(-0.03,0.03), R.uniform(-0.03,0.03)),
                       misalign=(math.radians(R.uniform(-0.7,0.7)), math.radians(R.uniform(-0.7,0.7))),
                       pad_tilt=(math.radians(R.uniform(-3,3)), math.radians(R.uniform(-3,3))),
                       gyro_bias=tuple(math.radians(R.uniform(-3,3)) for _ in range(3)), **kw))
    v = np.array([o["v"] for o in out]); tl = np.array([o["tilt"] for o in out]); vh = np.array([o["vh"] for o in out])
    ok = (v <= 1) & (tl <= 5) & (vh <= 0.5)
    good = [o["miss"] for o in out if o["miss"] < 99]
    return dict(ok=ok.mean()*100, miss=np.percentile(good, 95) if good else 99,
                rng=np.percentile([o["range"] for o in out], 95), vh=np.percentile([o["vh"] for o in out if o["vh"] < 99], 95))

if __name__ == "__main__":
    kw = dict(T_scale=0.9, tau=0.2, m_scale=1.03, gust=(0.6,-0.4), leg_kick=(0.02,-0.015),
              misalign=(math.radians(0.5), math.radians(-0.4)), pad_tilt=(math.radians(2), math.radians(-1.5)),
              gyro_bias=(math.radians(2), math.radians(-1.5), math.radians(2.5)))
    navs = [("cheat", "perfect position (steps 8-12)"), ("none", "no sideways info, hold level"),
            ("deadreckon", "dead reckoning (integrate accel)"), ("flow", "optical flow below 4 m"),
            ("gps", "GPS, 10 Hz, 1.5 m bias")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    print("same detailed flight, five navigation options:")
    for nav, label in navs:
        res, ev, rec = fly(3, nav=nav, log=True, **kw)
        x_ = [p[1] for p in rec]; y_ = [p[2] for p in rec]; z_ = [p[3] for p in rec]
        ax[0].plot(x_, y_, label=label); ax[0].plot(x_[-1], y_[-1], "o")
        ax[1].plot([p[0] for p in rec], [math.hypot(a, b) for a, b in zip(x_, y_)], label=label)
        print(f"  {label:35s} touchdown {res['v']:5.2f} m/s down, {res['vh']:5.2f} m/s sideways, "
              f"{res['miss']:6.2f} m from pad, max {res['range']:6.1f} m away")
    ax[0].plot(0, 0, "k*", ms=15); ax[0].set_title("top view (star = pad)"); ax[0].set_xlabel("x, m"); ax[0].set_ylabel("y, m")
    ax[0].axis("equal"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].set_title("distance from pad"); ax[1].set_xlabel("time, s"); ax[1].set_ylabel("m"); ax[1].grid(alpha=0.3)
    ax[1].set_yscale("symlog", linthresh=1)
    plt.tight_layout(); plt.savefig("step13.png", dpi=100)

    print("\nMonte Carlo, 80 runs each. Success = down <= 1 m/s, tilt <= 5 deg, sideways <= 0.5 m/s")
    print("  navigation                          | success | 95% miss | 95% max range | 95% sideways speed")
    navs += [("cheat+", "perfect position + terminal v-null"), ("gps+", "GPS + terminal v-null"),
             ("gps+flow+", "GPS + flow + terminal v-null")]
    for nav, label in navs:
        tn = nav.endswith("+") and nav != "gps+flow"; nv = nav.rstrip("+") if tn else nav
        if nav == "gps+flow+": nv, tn = "gps+flow", True
        s = mc(nav=nv, term_null=tn)
        print(f"  {label:36s}|  {s['ok']:4.0f}%  | {s['miss']:6.1f} m | {s['rng']:7.1f} m     | {s['vh']:5.2f} m/s")
    plt.show()
