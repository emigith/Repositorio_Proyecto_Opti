import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

os.makedirs("figures_fase3", exist_ok=True)

# =========================
# Parámetros del modelo corregido
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

# =========================
# Equilibrios de la Fase 2
# =========================

P1 = np.array([697.22, 0.0, 0.0])
P2 = np.array([100.0, 20.2448, 8676.35])
P3 = np.array([250.0, 6.0640, 1143.50])

equilibrios = {
    "P1": P1,
    "P2": P2,
    "P3": P3
}

clasificacion = {
    "P1": "Nodo atractor local",
    "P2": "Nodo atractor local",
    "P3": "Punto silla"
}

# =========================
# Sistema no lineal corregido
# =========================

def sistema_corregido(t, x):
    G, I, beta = x

    dG = p["R0"] + p["Ge"] - (p["EGO"] + (1 - p["rhoR"]) * p["SI"] * I) * G

    dI = (1 - p["rhoS"]) * (
        beta * p["sigma"] * G**2 / (p["alpha"] + G**2)
    ) - p["gamma"] * I

    dbeta = (-p["d0"] + p["r1"] * G - p["r2"] * G**2) * beta

    return np.array([dG, dI, dbeta])

# =========================
# Jacobiano corregido
# =========================

def jacobiano(G, I, beta):
    R0 = p["R0"]
    Ge = p["Ge"]
    EGO = p["EGO"]
    SI = p["SI"]
    sigma = p["sigma"]
    alpha = p["alpha"]
    gamma = p["gamma"]
    rhoR = p["rhoR"]
    rhoS = p["rhoS"]
    d0 = p["d0"]
    r1 = p["r1"]
    r2 = p["r2"]

    a = (1 - rhoR) * SI
    b = (1 - rhoS)

    J = np.array([
        [
            -(EGO + a * I),
            -a * G,
            0
        ],
        [
            b * beta * sigma * (2 * alpha * G) / (alpha + G**2)**2,
            -gamma,
            b * sigma * G**2 / (alpha + G**2)
        ],
        [
            (r1 - 2 * r2 * G) * beta,
            0,
            -d0 + r1 * G - r2 * G**2
        ]
    ])

    return J

# =========================
# Imprimir valores propios
# =========================

print("Clasificación local mediante valores propios:\n")

for nombre, P in equilibrios.items():
    J = jacobiano(P[0], P[1], P[2])
    eigvals, eigvecs = np.linalg.eig(J)

    print(nombre)
    print("Punto:", P)
    print("Jacobiano:")
    print(J)
    print("Valores propios:")
    print(eigvals)
    print("Clasificación:", clasificacion[nombre])
    print("-" * 50)

# =========================
# Bosquejos locales en planos coordenados
# =========================

def campo_local_2D(nombre, P, indices, etiquetas, archivo, escala_x, escala_y):
    """
    Grafica el campo linealizado local en un plano coordenado.
    Se trabaja con variables de desviación:
    u = x - P.
    """

    J = jacobiano(P[0], P[1], P[2])

    A = J[np.ix_(indices, indices)]

    x = np.linspace(-escala_x, escala_x, 25)
    y = np.linspace(-escala_y, escala_y, 25)
    X, Y = np.meshgrid(x, y)

    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y

    norma = np.sqrt(U**2 + V**2)
    norma[norma == 0] = 1

    U_n = U / norma
    V_n = V / norma

    plt.figure(figsize=(7, 6))
    plt.quiver(X, Y, U_n, V_n, color="gray", alpha=0.75)

    # Trayectorias del sistema linealizado 2D
    def sistema_lineal_2d(t, z):
        return A @ z

    condiciones = [
        [escala_x, escala_y],
        [-escala_x, escala_y],
        [escala_x, -escala_y],
        [-escala_x, -escala_y],
        [escala_x, 0],
        [-escala_x, 0],
        [0, escala_y],
        [0, -escala_y],
        [0.5 * escala_x, 0.5 * escala_y],
        [-0.5 * escala_x, 0.5 * escala_y]
    ]

    t_span = (0, 0.1)

    # Para P3 también graficamos hacia atrás para ver mejor las direcciones de silla
    if nombre == "P3":
        t_span = (0, 2)

    for z0 in condiciones:
        sol = solve_ivp(
            sistema_lineal_2d,
            t_span,
            z0,
            t_eval=np.linspace(t_span[0], t_span[1], 300)
        )

        plt.plot(sol.y[0], sol.y[1], linewidth=2)

        # Flecha sobre la trayectoria
        if sol.y.shape[1] > 20:
            idx1 = int(0.55 * sol.y.shape[1])
            idx2 = int(0.65 * sol.y.shape[1])
            plt.annotate(
                "",
                xy=(sol.y[0, idx2], sol.y[1, idx2]),
                xytext=(sol.y[0, idx1], sol.y[1, idx1]),
                arrowprops=dict(arrowstyle="->", lw=1.5)
            )

    plt.scatter(0, 0, s=120, color="red", zorder=5)
    plt.text(0.03 * escala_x, 0.08 * escala_y, nombre, fontsize=13, weight="bold")

    plt.title(f"Bosquejo local alrededor de {nombre}\n{clasificacion[nombre]}")
    plt.xlabel(etiquetas[0])
    plt.ylabel(etiquetas[1])
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    plt.savefig(f"figures_fase3/{archivo}.png", dpi=300)
    plt.show()

# =========================
# Generar bosquejos locales
# =========================

for nombre, P in equilibrios.items():

    if nombre == "P1":
        escala_G = 30
        escala_I = 0.55
        escala_B = 200
    elif nombre == "P2":
        escala_G = 20
        escala_I = 5
        escala_B = 1500
    else:
        escala_G = 30
        escala_I = 4
        escala_B = 600

    # Plano G-I
    campo_local_2D(
        nombre,
        P,
        indices=[0, 1],
        etiquetas=[r"$u_G = G-G^*$", r"$u_I = I-I^*$"],
        archivo=f"{nombre}_local_GI",
        escala_x=escala_G,
        escala_y=escala_I
    )

    # Plano G-beta
    campo_local_2D(
        nombre,
        P,
        indices=[0, 2],
        etiquetas=[r"$u_G = G-G^*$", r"$u_\beta = \beta-\beta^*$"],
        archivo=f"{nombre}_local_Gbeta",
        escala_x=escala_G,
        escala_y=escala_B
    )

    # Plano I-beta
    campo_local_2D(
        nombre,
        P,
        indices=[1, 2],
        etiquetas=[r"$u_I = I-I^*$", r"$u_\beta = \beta-\beta^*$"],
        archivo=f"{nombre}_local_Ibeta",
        escala_x=escala_I,
        escala_y=escala_B
    )

# =========================
# Simulaciones no lineales alrededor de cada equilibrio
# =========================

def simular_alrededor_equilibrio(nombre, P, perturbaciones, t_final, archivo):
    plt.figure(figsize=(9, 7))
    ax = plt.axes(projection="3d")

    for pert in perturbaciones:
        x0 = P + np.array(pert)

        # Evitar condiciones iniciales negativas
        x0 = np.maximum(x0, 0)

        sol = solve_ivp(
            sistema_corregido,
            (0, t_final),
            x0,
            method="BDF",
            t_eval=np.linspace(0, t_final, 2000),
            rtol=1e-8,
            atol=1e-10
        )

        ax.plot(sol.y[0], sol.y[1], sol.y[2], linewidth=2)

    ax.scatter(P[0], P[1], P[2], color="red", s=80, label=nombre)

    ax.set_title(f"Simulaciones no lineales alrededor de {nombre}\n{clasificacion[nombre]}")
    ax.set_xlabel("G")
    ax.set_ylabel("I")
    ax.set_zlabel(r"$\beta$")
    ax.legend()
    plt.tight_layout()

    plt.savefig(f"figures_fase3/{archivo}.png", dpi=300)
    plt.show()

# Perturbaciones locales
perturbaciones_P1 = [
    [20, 1, 20],
    [-20, 1, 20],
    [40, 2, 50],
    [-40, 2, 50],
    [10, 0.5, 100]
]

perturbaciones_P2 = [
    [5, 1, 200],
    [-5, 1, 200],
    [10, -1, -200],
    [-10, -1, -200],
    [3, 0.5, -500]
]

perturbaciones_P3 = [
    [5, 0.5, 100],
    [-5, 0.5, 100],
    [10, -0.5, -100],
    [-10, -0.5, -100],
    [3, 0.2, 300],
    [-3, -0.2, -300]
]

simular_alrededor_equilibrio(
    "P1",
    P1,
    perturbaciones_P1,
    t_final=30,
    archivo="simulacion_local_P1"
)

simular_alrededor_equilibrio(
    "P2",
    P2,
    perturbaciones_P2,
    t_final=100,
    archivo="simulacion_local_P2"
)

simular_alrededor_equilibrio(
    "P3",
    P3,
    perturbaciones_P3,
    t_final=150,
    archivo="simulacion_local_P3"
)

# =========================
# Simulación global en el espacio de estados
# =========================

condiciones_globales = [
    [0, 10, 300],
    [100, 0, 300],
    [80, 10, 1000],
    [120, 25, 9000],
    [200, 8, 2000],
    [260, 6, 1200],
    [400, 4, 500],
    [700, 1, 100],
    [50, 30, 10000],
    [300, 10, 3000]
]

plt.figure(figsize=(10, 8))
ax = plt.axes(projection="3d")

for x0 in condiciones_globales:
    sol = solve_ivp(
        sistema_corregido,
        (0, 400),
        x0,
        method="BDF",
        t_eval=np.linspace(0, 400, 4000),
        rtol=1e-8,
        atol=1e-10
    )

    ax.plot(sol.y[0], sol.y[1], sol.y[2], linewidth=2)

# Marcar equilibrios
ax.scatter(P1[0], P1[1], P1[2], color="red", s=100, label="P1 estable patológico")
ax.scatter(P2[0], P2[1], P2[2], color="blue", s=100, label="P2 estable fisiológico")
ax.scatter(P3[0], P3[1], P3[2], color="black", s=100, label="P3 silla")

ax.set_title("Simulación global del sistema corregido en el espacio de estados")
ax.set_xlabel("G")
ax.set_ylabel("I")
ax.set_zlabel(r"$\beta$")
ax.legend()
plt.tight_layout()

plt.savefig("figures_fase3/simulacion_global_3D.png", dpi=300)
plt.show()

# =========================
# Proyección global en G-I
# =========================

plt.figure(figsize=(8, 6))

for x0 in condiciones_globales:
    sol = solve_ivp(
        sistema_corregido,
        (0, 400),
        x0,
        method="BDF",
        t_eval=np.linspace(0, 400, 4000),
        rtol=1e-8,
        atol=1e-10
    )

    plt.plot(sol.y[0], sol.y[1], linewidth=2)

plt.scatter(P1[0], P1[1], color="red", s=100, label="P1")
plt.scatter(P2[0], P2[1], color="blue", s=100, label="P2")
plt.scatter(P3[0], P3[1], color="black", s=100, label="P3")

plt.xlabel("G")
plt.ylabel("I")
plt.title("Proyección global en el plano G-I")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig("figures_fase3/proyeccion_global_GI.png", dpi=300)
plt.show()

# =========================
# Proyección global en G-beta
# =========================

plt.figure(figsize=(8, 6))

for x0 in condiciones_globales:
    sol = solve_ivp(
        sistema_corregido,
        (0, 400),
        x0,
        method="BDF",
        t_eval=np.linspace(0, 400, 4000),
        rtol=1e-8,
        atol=1e-10
    )

    plt.plot(sol.y[0], sol.y[2], linewidth=2)

plt.scatter(P1[0], P1[2], color="red", s=100, label="P1")
plt.scatter(P2[0], P2[2], color="blue", s=100, label="P2")
plt.scatter(P3[0], P3[2], color="black", s=100, label="P3")

plt.xlabel("G")
plt.ylabel(r"$\beta$")
plt.title(r"Proyección global en el plano $G-\beta$")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig("figures_fase3/proyeccion_global_Gbeta.png", dpi=300)
plt.show()


import shutil
shutil.make_archive("figures_fase3", "zip", "figures_fase3")