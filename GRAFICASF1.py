import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================
# PARAMETROS
# =========================
p = {
    "R0": 864,
    "Ge": 140,
    "EGO": 1.44,
    "SI": 0.72,
    "sigma": 43.2,
    "alpha": 20000,
    "gamma": 432,
    "rhoR": 0.41,
    "rhoS": 0.93,
    "d0": 0.06,
    "r1": 0.84e-3,
    "r2": 0.24e-5
}

# Equilibrio degenerado
G_star = (p["R0"] + p["Ge"]) / p["EGO"]
E0 = np.array([G_star, 0, 0])
print("Equilibrio degenerado E0 =", E0)

# =========================
# MODELO CORREGIDO
# =========================
def modelo_corregido(t, x, p):
    G, I, B = x

    dG = p["R0"] + p["Ge"] - (p["EGO"] + (1 - p["rhoR"]) * p["SI"] * I) * G
    dI = (1 - p["rhoS"]) * (B * p["sigma"] * G**2 / (p["alpha"] + G**2)) - p["gamma"] * I
    dB = (-p["d0"] + p["r1"] * G - p["r2"] * G**2) * B

    return [dG, dI, dB]

# =========================
# SIMULACIONES
# =========================
t_span = (0, 400)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Caso F1.1: G(0)=0
x0_case1 = [0, 10, 300]
sol1 = solve_ivp(modelo_corregido, t_span, x0_case1, args=(p,),
                 method='BDF', t_eval=t_eval)

# Caso F1.2: I(0)=0
x0_case2 = [100, 0, 300]
sol2 = solve_ivp(modelo_corregido, t_span, x0_case2, args=(p,),
                 method='BDF', t_eval=t_eval)

# =========================
# GRAFICAS TEMPORALES
# =========================
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

# Caso 1
axs[0,0].plot(sol1.t, sol1.y[0], 'b', lw=2)
axs[0,0].set_title('Caso F1.1: G(0)=0')
axs[0,0].set_ylabel('G(t)')
axs[0,0].grid(True)

axs[1,0].plot(sol1.t, sol1.y[1], 'r', lw=2)
axs[1,0].set_title('Insulina - Caso F1.1')
axs[1,0].set_ylabel('I(t)')
axs[1,0].grid(True)

axs[2,0].plot(sol1.t, sol1.y[2], 'g', lw=2)
axs[2,0].set_title('Células beta - Caso F1.1')
axs[2,0].set_ylabel('B(t)')
axs[2,0].set_xlabel('t')
axs[2,0].grid(True)

# Caso 2
axs[0,1].plot(sol2.t, sol2.y[0], 'b', lw=2)
axs[0,1].set_title('Caso F1.2: I(0)=0')
axs[0,1].set_ylabel('G(t)')
axs[0,1].grid(True)

axs[1,1].plot(sol2.t, sol2.y[1], 'r', lw=2)
axs[1,1].set_title('Insulina - Caso F1.2')
axs[1,1].set_ylabel('I(t)')
axs[1,1].grid(True)

axs[2,1].plot(sol2.t, sol2.y[2], 'g', lw=2)
axs[2,1].set_title('Células beta - Caso F1.2')
axs[2,1].set_ylabel('B(t)')
axs[2,1].set_xlabel('t')
axs[2,1].grid(True)

plt.tight_layout()
plt.show()

# =========================
# PLANO FASE (G,I)
# =========================
plt.figure(figsize=(8,6))
plt.plot(sol1.y[0], sol1.y[1], 'b', lw=2, label='Caso F1.1: G(0)=0')
plt.plot(sol2.y[0], sol2.y[1], 'r', lw=2, label='Caso F1.2: I(0)=0')
plt.scatter(E0[0], E0[1], c='y', edgecolors='k', s=100, label='E0')
plt.xlabel('G')
plt.ylabel('I')
plt.title('Trayectorias en el plano (G,I)')
plt.grid(True)
plt.legend()
plt.show()

# =========================
# CAMPO DIRECCIONAL EN (G,I)
# fijando B = Bfix
# =========================
Bfix = 300
G_vals = np.linspace(0, 800, 20)
I_vals = np.linspace(0, 80, 20)
Gm, Im = np.meshgrid(G_vals, I_vals)

dG = p["R0"] + p["Ge"] - (p["EGO"] + (1 - p["rhoR"]) * p["SI"] * Im) * Gm
dI = (1 - p["rhoS"]) * (Bfix * p["sigma"] * Gm**2 / (p["alpha"] + Gm**2)) - p["gamma"] * Im

N = np.sqrt(dG**2 + dI**2)
dG_n = dG / (N + 1e-9)
dI_n = dI / (N + 1e-9)

plt.figure(figsize=(8,6))
plt.quiver(Gm, Im, dG_n, dI_n, color='k', alpha=0.7)
plt.plot(sol1.y[0], sol1.y[1], 'b', lw=2, label='Caso F1.1')
plt.plot(sol2.y[0], sol2.y[1], 'r', lw=2, label='Caso F1.2')
plt.scatter(E0[0], E0[1], c='y', edgecolors='k', s=100, label='E0')
plt.xlabel('G')
plt.ylabel('I')
plt.title(f'Campo direccional en (G,I) con B fijo = {Bfix}')
plt.grid(True)
plt.legend()
plt.show()

# =========================
# TRAYECTORIA 3D
# =========================
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol1.y[0], sol1.y[1], sol1.y[2], 'b', lw=2, label='Caso F1.1')
ax.plot(sol2.y[0], sol2.y[1], sol2.y[2], 'r', lw=2, label='Caso F1.2')
ax.scatter(E0[0], E0[1], E0[2], c='y', edgecolors='k', s=100, label='E0')
ax.set_xlabel('G')
ax.set_ylabel('I')
ax.set_zlabel('B')
ax.set_title('Trayectorias 3D del sistema corregido')
ax.legend()
plt.show()