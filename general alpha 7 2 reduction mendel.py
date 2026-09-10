import numpy as np
import matplotlib.pyplot as plt
import time

# تنظیم دانه تصادفی
np.random.seed(42)

# ============================================================
# ۱. پارامترهای فیزیکی سیستم تعلیق
# ============================================================
ms, mus = 500, 40
ks, kt = 10000, 150000
cs = 1000
e_max, de_max, f_max = 0.1, 1, 900

# پارامترهای ۷ گانه توابع عضویت
mf_params = {
    'NB': [-1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25],
    'NS': [-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25],
    'ZE': [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75],
    'PS': [-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25],
    'PB': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]
}


# ============================================================
# ۲. توابع عضویت و موتور استنتاج (بهینه شده برای سرعت)
# ============================================================

def get_alpha_membership_fast(x, p, alpha):
    """ محاسبه سریع بازه عضویت بدون استفاده از Numpy برای افزایش سرعت اجرا """
    a, b, c, m, c2, b2, a2 = p
    t1 = t2 = t3 = 0.0

    # محاسبات منطبق بر کد اصلاح شده قبلی
    if x < m:
        if x > a: t1 = (x - a) / (m - a)
        if x > b: t2 = 0.2 * (x - a) / (m - a) + 0.8 * 0.5 * (x - c) / (m - c)
        if x > c: t3 = 0.5 * (x - c) / (m - c)
    else:
        if x < a2: t1 = (a2 - x) / (a2 - m)
        if x < b2: t2 = 0.2 * (a2 - x) / (a2 - m) + 0.8 * 0.5 * (c2 - x) / (c2 - m)
        if x < c2: t3 = 0.5 * (c2 - x) / (c2 - m)

    # محاسبه کران بالا و پایین (فرمول پیشنهادی شما)
    u = max(0.0, min(1.0, t1 - alpha * (t1 - t2)))
    l = max(0.0, min(1.0, t3 + alpha * (t2 - t3)))
    return l, u


# پارامترهای تستی برای یک تابع عضویت (مثلاً ZE)
p_test = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
x_range = np.linspace(-1, 1, 500)
alphas = [0,1]  # سطوح مختلف برای مشاهده انقباض
colors = ['red', 'green', 'blue']

plt.figure(figsize=(10, 6))

for alpha, col in zip(alphas, colors):
    l_vals = []
    u_vals = []
    for x in x_range:
        l, u = get_alpha_membership_fast(x, p_test, alpha)
        l_vals.append(l)
        u_vals.append(u)

    # رسم ناحیه بین L و U (FOU)
    plt.fill_between(x_range, l_vals, u_vals, color=col, alpha=0.3, label=f'Alpha = {alpha}')
    plt.plot(x_range, u_vals, color=col, linestyle='--', linewidth=1)
    plt.plot(x_range, l_vals, color=col, linewidth=1.5)

plt.title("Visual Validation of Contracting GT2 Membership Function")
plt.xlabel("Normalized Input")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(0, color='black', alpha=0.5)  # مرکز (m)
plt.show()


def type2_inference(e_val, de_val):
    #e_in = max(-1.0, min(1.0, e_val / e_max))+np.random.normal(0, 0.08)
    #de_in = max(-1.0, min(1.0, de_val / de_max))+np.random.normal(0, 0.07)

    e_in = max(-1.0, min(1.0, e_val / e_max + np.random.normal(0, 0.08)))
    de_in = max(-1.0, min(1.0, de_val / de_max + np.random.normal(0, 0.07)))


    # استفاده از ۱۵ سطح آلفا برای تعادل بین سرعت و دقت
    alpha_levels = np.linspace(0.1, 1.0, 10)

    y_sum_num, y_sum_den = 0.0, 0.0
    labels = ['NB', 'ZE', 'PB']
    rule_table = [['PB', 'PS', 'ZE'], ['PS', 'ZE', 'NS'], ['ZE', 'NS', 'NB']]

    # global interval aggregation over all alpha-levels
    glob_num_l = 0.0
    glob_den_l = 0.0
    glob_num_u = 0.0
    glob_den_u = 0.0

    for alpha in alpha_levels:

        for i, e_lab in enumerate(labels):
            for j, de_lab in enumerate(labels):
                out_lab = rule_table[i][j]

                l_e, u_e = get_alpha_membership_fast(
                    e_in, mf_params[e_lab], alpha
                )
                l_de, u_de = get_alpha_membership_fast(
                    de_in, mf_params[de_lab], alpha
                )

                # firing interval
                f_l = min(l_e, l_de)
                f_u = min(u_e, u_de)

                c_m = mf_params[out_lab][3]

                # aggregate globally across alpha
                glob_num_l += alpha * f_l * c_m
                glob_den_l += alpha * f_l

                glob_num_u += alpha * f_u * c_m
                glob_den_u += alpha * f_u

    # -------------------------------
    # single type-reduction (ONLY ONCE)
    # -------------------------------
    if glob_den_l > 1e-6 and glob_den_u > 1e-6:
        Y_l = glob_num_l / glob_den_l
        Y_u = glob_num_u / glob_den_u

        # final crisp output
        y_crisp = 0.5 * (Y_l + Y_u) * f_max
    else:
        y_crisp = 0.0

    return y_crisp


# ============================================================
# ۳. مدل دینامیکی و حلگر RK4
# ============================================================

def road_profile(t):
    if 1.0 <= t <= 1.2:
        return 0.05 * np.sin(np.pi * (t - 1.0) / 0.2)
    elif 5.0 <= t <= 5.5:
        return -0.01 * np.sin(np.pi * (t - 1.0) / 0.2)
    elif 5.5 <= t <= 6.5:
        return -0.07 * np.sin(np.pi * (t-0.7) / 0.1)
    return 0



def suspension_ode(t, y, fa):
    xs, dxs, xus, dxus = y
    xr = road_profile(t)
    ddxs = (ks * (xus - xs) + cs * (dxus - dxs) + fa) / ms
    ddxus = (ks * (xs - xus) + cs * (dxs - dxus) + kt * (xr - xus) - fa) / mus
    return np.array([dxs, ddxs, dxus, ddxus])


# شبیه‌سازی
dt, T_end = 0.01, 20.0
time_steps = np.arange(0, T_end, dt)
state = np.zeros(4)
res_xs, res_acc, res_fa = [], [], []

print("Simulating... Please wait.")
start_t = time.time()

for t in time_steps:
    fa = type2_inference(state[0], state[1] - state[3])

    # رانگ-کوتا مرتبه ۴
    k1 = suspension_ode(t, state, fa)
    k2 = suspension_ode(t + dt / 2, state + dt * k1 / 2, fa)
    k3 = suspension_ode(t + dt / 2, state + dt * k2 / 2, fa)
    k4 = suspension_ode(t + dt, state + dt * k3, fa)
    state += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    res_xs.append(state[0])
    res_acc.append(k1[1])
    res_fa.append(fa)

end_t = time.time()

# ============================================================
# ۴. استخراج شاخص‌های آماری پیشرفته
# ============================================================
res_acc = np.array(res_acc)
res_xs = np.array(res_xs)
res_f = np.array(res_fa)

iae_acc = np.trapz(np.abs(res_acc), time_steps)
itae_acc = np.trapz(time_steps * np.abs(res_acc), time_steps)
iae_xs = np.trapz(np.abs(res_xs), time_steps)
itae_xs = np.trapz(time_steps * np.abs(res_xs), time_steps)

metrics = {
    'RMS': [np.sqrt(np.mean(np.square(res_acc))), np.sqrt(np.mean(np.square(res_xs))),
            np.sqrt(np.mean(np.square(res_f)))],
    'Peak': [np.max(np.abs(res_acc)), np.max(np.abs(res_xs)), np.max(np.abs(res_f))]
}

print(f"\nSimulation Time: {end_t - start_t:.2f} seconds")
print(f"\n{'=' * 40}\n   PERFORMANCE METRICS (ADVANCED)\n{'=' * 40}")
print(f"Sprung Acc (RMS):   {metrics['RMS'][0]:.4f} m/s²")
print(f"Sprung Acc (Peak):  {metrics['Peak'][0]:.4f} m/s²")
print(f"Sprung Acc (IAE):   {iae_acc:.4f}")
print(f"Sprung Acc (ITAE):  {itae_acc:.4f}")
print(f"{'-' * 40}")
print(f"Body Disp (RMS):    {metrics['RMS'][1]:.4f} m")
print(f"Body Disp (Peak):   {metrics['Peak'][1]:.4f} m")
print(f"Body Disp (IAE):    {iae_xs:.4f}")
print(f"Body Disp (ITAE):   {itae_xs:.4f}")
print(f"{'-' * 40}")
print(f"Force (RMS):       {metrics['RMS'][2]:.2f} N")
print(f"Force (Peak):      {metrics['Peak'][2]:.2f} N")
print(f"{'=' * 40}")

# رسم نمودار نهایی
plt.figure(figsize=(10, 5))
plt.plot(time_steps, res_acc, label='Body Acceleration (Comfort)', color='red')
plt.grid(True);
plt.legend();
plt.xlabel('Time (s)');
plt.show()