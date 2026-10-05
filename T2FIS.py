import numpy as np
import time

_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

np.random.seed(42)

ms, mus = 500, 40
ks, kt = 10000, 150000
cs = 1000
e_max, de_max, f_max = 0.025, 3.5, 900

ks_nl = 0.3 * ks

mf_params = {
    'NB': [-1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25],
    'NS': [-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25],
    'ZE': [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75],
    'PS': [-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25],
    'PB': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]
}



def get_alpha_membership_fast(x, p, alpha):
    a, b, c, m, c2, b2, a2 = p
    t1 = t2 = t3 = 0.0
    if x < m:
        if x > a:
            t1 = (x - a) / (m - a)
        if x > b:
            t2 = (0.2 * (x - a) / (m - a)+ 0.8 * 0.5 * (x - c) / (m - c))
        if x > c:t3 = 0.5 * (x - c) / (m - c)
    else:
        if x < a2:
            t1 = (a2 - x) / (a2 - m)
        if x < b2:
            t2 = (0.2 * (a2 - x) / (a2 - m)+ 0.8 * 0.5 * (c2 - x) / (c2 - m))
        if x < c2:
            t3 = 0.5 * (c2 - x) / (c2 - m)
    u = max(0.0,min(1.0,t1 - alpha * (t1 - t2)))
    l = max(0.0,min(1.0,t3 + alpha * (t2 - t3)))
    return l, u

def type2_inference(e_val, de_val):
    e_in = max(-1.0,min(1.0,e_val / e_max + np.random.normal(0, 0.08)))
    de_in = max(-1.0,min(1.0,de_val / de_max + np.random.normal(0, 0.07)))

    alpha_levels = np.linspace(0.1, 1.0, 10)
    labels = ['NB', 'ZE', 'PB']
    rule_table = [
        ['PB', 'PS', 'ZE'],
        ['PS', 'ZE', 'NS'],
        ['ZE', 'NS', 'NB']
    ]
    alpha_weighted_y_l = 0.0
    alpha_weighted_y_u = 0.0

    alpha_sum_l = 0.0
    alpha_sum_u = 0.0

    for alpha in alpha_levels:
        num_l_alpha = 0.0
        den_l_alpha = 0.0

        num_u_alpha = 0.0
        den_u_alpha = 0.0
        for i, e_lab in enumerate(labels):
            for j, de_lab in enumerate(labels):
                out_lab = rule_table[i][j]
                l_e, u_e = get_alpha_membership_fast(e_in,mf_params[e_lab],alpha)
                l_de, u_de = get_alpha_membership_fast(de_in,mf_params[de_lab],alpha)
                f_l = min(l_e, l_de)
                f_u = min(u_e, u_de)
                c_m = mf_params[out_lab][3]
                num_l_alpha += f_l * c_m
                den_l_alpha += f_l
                num_u_alpha += f_u * c_m
                den_u_alpha += f_u
        if den_l_alpha > 1e-6 and den_u_alpha > 1e-6:
            y_l_alpha = (num_l_alpha / den_l_alpha)
            y_u_alpha = (num_u_alpha / den_u_alpha)
        else:
            continue
        alpha_weighted_y_l += alpha * y_l_alpha
        alpha_weighted_y_u += alpha * y_u_alpha

        alpha_sum_l += alpha
        alpha_sum_u += alpha

    if alpha_sum_l > 1e-12 and alpha_sum_u > 1e-12:
        Y_l = (alpha_weighted_y_l/ alpha_sum_l)
        Y_u = (alpha_weighted_y_u/ alpha_sum_u)
        y_crisp = (0.5 * (Y_l + Y_u) * f_max)
    else:
        y_crisp = 0.0
    return y_crisp

def road_profile(t):
    if 1.0 <= t <= 1.2:
        return (0.05* np.sin( np.pi * (t - 1.0) / 0.2))
    elif 5.0 <= t <= 5.5:
        return (-0.01* np.sin(np.pi * (t - 1.0) / 0.2))
    elif 5.5 <= t <= 6.5:
        return (-0.07* np.sin(np.pi * (t - 0.7) / 0.1))
    return 0

def spring_force(dx):
    return (ks * dx+ ks_nl * dx ** 3)

def suspension_ode(t, y, fa):
    xs, dxs, xus, dxus = y
    xr = road_profile(t)
    ddxs = (spring_force(xus - xs)+ cs * (dxus - dxs)+ fa) / ms
    ddxus = (spring_force(xs - xus)+ cs * (dxs - dxus)+ kt * (xr - xus)- fa) / mus
    return np.array([dxs,ddxs,dxus,ddxus])

dt = 0.01
T_end = 20.0

time_steps = np.arange(0,T_end,dt)
state = np.zeros(4)
res_xs = []
res_acc = []
res_fa = []

start_t = time.perf_counter()
for t in time_steps:
    fa = type2_inference(state[0],state[1] - state[3])
    k1 = suspension_ode(t,state,fa)
    k2 = suspension_ode(t + dt / 2,state + dt * k1 / 2,fa)
    k3 = suspension_ode(t + dt / 2,state + dt * k2 / 2,fa)
    k4 = suspension_ode(t + dt,state + dt * k3,fa)
    state += (dt / 6* (k1+ 2 * k2+ 2 * k3+ k4))
    res_xs.append(state[0])
    res_acc.append(k1[1])
    res_fa.append(fa)
end_t = time.perf_counter()
res_acc = np.array(res_acc)
res_xs = np.array(res_xs)
res_f = np.array(res_fa)

iae_acc = _trapz(np.abs(res_acc),time_steps)
itae_acc = _trapz(time_steps * np.abs(res_acc),time_steps)
iae_xs = _trapz(np.abs(res_xs),time_steps)
itae_xs = _trapz(time_steps * np.abs(res_xs),time_steps)

metrics = {'RMS': [np.sqrt(np.mean(np.square(res_acc))),np.sqrt(np.mean(np.square(res_xs))),np.sqrt(np.mean(np.square(res_f)))],
    'Peak': [np.max(np.abs(res_acc)),np.max(np.abs(res_xs)),np.max(np.abs(res_f))]}

print(f"\nSimulation Time: "f"{end_t - start_t:.2f} seconds")
print(f"\n{'=' * 40}\n"f"   Mendel (T2FIS) \n"f"{'=' * 40}")
print(f"Sprung Acc (RMS):   "f"{metrics['RMS'][0]:.4f} m/s²")
print(f"Sprung Acc (Peak):  "f"{metrics['Peak'][0]:.4f} m/s²")
print(f"Sprung Acc (IAE):   "f"{iae_acc:.4f}")
print(f"Sprung Acc (ITAE):  "f"{itae_acc:.4f}")
print(f"{'-' * 40}")
print(f"Body Disp (RMS):    "f"{metrics['RMS'][1]:.4f} m")
print(f"Body Disp (Peak):   "f"{metrics['Peak'][1]:.4f} m")
print(f"Body Disp (IAE):    "f"{iae_xs:.4f}")
print(f"Body Disp (ITAE):   "f"{itae_xs:.4f}")
print(f"{'-' * 40}")
print(f"Force (RMS):       "f"{metrics['RMS'][2]:.2f} N")
print(f"Force (Peak):      "f"{metrics['Peak'][2]:.2f} N")
print(f"{'=' * 40}")
