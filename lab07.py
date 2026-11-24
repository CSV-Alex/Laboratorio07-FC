# Ejercicio 1) Rendija simple rectangular y comparación con sinc^2.
# Ejercicio 2) Doble rendija y red de 5 rendijas.
# Ejercicio 3) Apertura circular (Disco de Airy) y verificación de Parseval.
# Ejercicio 4) Propagación con ASM y comparación con Fraunhofer.

import numpy as np
import matplotlib.pyplot as plt

# Funciones generales

def malla(N, L):
    """
    Crea la malla espacial cuadrada de N x N puntos para un lado L.
    Devuelve X, Y y el paso dxp = L/N.
    """
    dxp = L / N
    x = np.linspace(-L/2, L/2 - dxp, N)
    X, Y = np.meshgrid(x, x)
    return X, Y, dxp


def campo_fraunhofer(A, dxp, lam, z):
    """
    Calcula el campo de Fraunhofer con FFT2D.
    A es la transmitancia (matriz NxN).
    dxp es el paso en el plano de apertura.
    lam es la longitud de onda y z la distancia a la pantalla.

    La FFT se centra con ifftshift antes de fft2 y se vuelve a centrar
    con fftshift después. El factor dxp^2 viene de aproximar la integral
    de Fourier por una suma discreta.
    """
    N = A.shape[0]
    L = N * dxp

    # Centrado de la apertura para que el orden cero quede al centro.
    A_shift = np.fft.ifftshift(A)

    # Transformada discreta 2D con el escalamiento de la integral.
    F = np.fft.fft2(A_shift) * dxp**2
    U = np.fft.fftshift(F)

    # Frecuencias espaciales centradas.
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dxp))
    x = lam * z * fx

    # Resolución espacial en la pantalla.
    dx = lam * z / L
    return U, x, dx


def intensidad_normalizada(U):
    """
    Devuelve |U|^2 normalizado por su máximo.
    Sirve para comparar perfiles sin depender de constantes globales.
    """
    I = np.abs(U)**2
    return I / np.max(I)


def campo_asm(A, dxp, lam, z):
    """
    Propaga el campo con el Método del Espectro Angular.
    Se usa la función de transferencia:
        H = exp(i z sqrt(k^2 - (2pi fx)^2 - (2pi fy)^2))
    Las componentes evanescentes se filtran poniendo H=0.
    """
    N = A.shape[0]
    k = 2 * np.pi / lam

    # Frecuencias sin centrar, tal como las usa fft2.
    fx = np.fft.fftfreq(N, d=dxp)
    FX, FY = np.meshgrid(fx, fx)

    # Argumento de la raíz. Si es negativo, la onda es evanescente.
    arg = k**2 - (2 * np.pi * FX)**2 - (2 * np.pi * FY)**2
    H = np.zeros_like(arg, dtype=complex)
    mascara = arg >= 0
    H[mascara] = np.exp(1j * z * np.sqrt(arg[mascara]))

    # Transformada, multiplicación por H y antitransformada.
    F = np.fft.fft2(A)
    U = np.fft.ifft2(F * H)

    # Se centra para graficar.
    U = np.fft.fftshift(U)

    # La malla de salida es la misma que la de entrada.
    x = np.linspace(-N*dxp/2, N*dxp/2 - dxp, N)
    return U, x


def minimo_cerca(x, perfil, x_teo, ventana):
    """
    Busca el mínimo local del perfil cerca de x_teo.
    Se usa para comparar la posición numérica del mínimo con la teórica.
    """
    idx = np.where(np.abs(x - x_teo) <= ventana)[0]
    if len(idx) == 0:
        return np.nan
    i_local = idx[np.argmin(perfil[idx])]
    return x[i_local]


# Ejercicio 1: rendija simple rectangular

def ejercicio1():
    print("\n--- Ejercicio 1: rendija simple rectangular ---")

    lam = 632.8e-9      # He-Ne
    a = 80e-6           # ancho de rendija
    z = 1.5             # distancia a pantalla
    L = 4e-3            # ventana de entrada
    N = 1024

    X, Y, dxp = malla(N, L)

    # Apertura rectangular centrada. La altura es suficientemente grande
    # para que el perfil horizontal no se vea afectado por los bordes.
    A = ((np.abs(X) <= a/2) & (np.abs(Y) <= L/4)).astype(float)

    U, x, dx = campo_fraunhofer(A, dxp, lam, z)
    I = intensidad_normalizada(U)
    perfil = I[N//2, :]

    # Perfil teórico: sinc^2(a x / (lam z)).
    # np.sinc(u) usa sin(pi u)/(pi u).
    I_teo = np.sinc(a * x / (lam * z))**2

    # Posiciones teóricas de los mínimos.
    x_teo = np.array([1, 2, 3]) * lam * z / a

    # Búsqueda de los mínimos numéricos cerca de los teóricos.
    x_num = np.array([minimo_cerca(x, perfil, xi, 5*dx) for xi in x_teo])
    error = np.abs(x_num - x_teo)

    print(f"dx' = {dxp:.6e} m")
    print(f"dx  = {dx:.6e} m")
    print("Mínimos teóricos [mm]:", x_teo * 1e3)
    print("Mínimos numéricos [mm]:", x_num * 1e3)
    print("Error absoluto [mm]:", error * 1e3)

    # Figura 1: mapa 2D y perfil transversal.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    im = ax[0].imshow(I, extent=[x[0]*1e3, x[-1]*1e3, x[0]*1e3, x[-1]*1e3],
                      origin="lower", cmap="inferno")
    ax[0].set_title("Intensidad 2D - rendija simple")
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("y [mm]")
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    ax[1].plot(x*1e3, perfil, "b-", label="FFT2D")
    ax[1].plot(x*1e3, I_teo, "r--", label=r"$\mathrm{sinc}^2(ax/\lambda z)$")
    ax[1].set_xlim(-50, 50)
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel(r"$I/I_0$")
    ax[1].set_title("Perfil transversal")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ej1_difraccion_rendija.png", dpi=200)
    plt.close()


# Ejercicio 2: doble rendija y red de 5 rendijas

def ejercicio2():
    print("\n--- Ejercicio 2: doble rendija y red de 5 rendijas ---")

    lam = 532e-9
    a = 60e-6
    d = 240e-6
    hy = 2e-3
    z = 2.0
    L = 5e-3
    N = 1024

    X, Y, dxp = malla(N, L)

    # Doble rendija: dos rectángulos centrados en -d/2 y +d/2.
    A2 = np.zeros_like(X)
    cond_x = (np.abs(X + d/2) <= a/2) | (np.abs(X - d/2) <= a/2)
    cond_y = np.abs(Y) <= hy/2
    A2[cond_x & cond_y] = 1

    # Red de 5 rendijas equiespaciadas.
    A5 = np.zeros_like(X)
    cond_x5 = np.zeros_like(X, dtype=bool)
    for c in (np.arange(5) - 2) * d:
        cond_x5 |= (np.abs(X - c) <= a/2)
    A5[cond_x5 & cond_y] = 1

    U2, x, dx = campo_fraunhofer(A2, dxp, lam, z)
    U5, _, _ = campo_fraunhofer(A5, dxp, lam, z)

    I2 = intensidad_normalizada(U2)
    I5 = intensidad_normalizada(U5)

    p2 = I2[N//2, :]
    p5 = I5[N//2, :]

    # Envolvente de rendija simple y teoría de doble rendija.
    envolvente = np.sinc(a * x / (lam * z))**2
    I2_teo = envolvente * np.cos(np.pi * d * x / (lam * z))**2

    print(f"Interfranja teórica: {lam*z/d*1e3:.4f} mm")
    print(f"Primer mínimo de envolvente: {lam*z/a*1e3:.4f} mm")
    print(f"d/a = {d/a:.2f}")

    # Figura 2a: geometría y perfil de la doble rendija.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].imshow(A2, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3],
                 origin="lower", cmap="gray")
    ax[0].set_title("Apertura: doble rendija")
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("y [mm]")

    ax[1].plot(x*1e3, p2, "b-", label="FFT2D")
    ax[1].plot(x*1e3, I2_teo, "r--", label="Teoría")
    ax[1].set_xlim(-50, 50)
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel(r"$I/I_0$")
    ax[1].set_title("Perfil de doble rendija")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ej2_doble_rendija.png", dpi=200)
    plt.close()

    # Figura 2b: comparación doble rendija vs red de 5.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].imshow(A5, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3],
                 origin="lower", cmap="gray")
    ax[0].set_title("Apertura: red de 5 rendijas")
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("y [mm]")

    ax[1].plot(x*1e3, p2, "b-", label="Doble rendija")
    ax[1].plot(x*1e3, p5, "g-", label="Red M=5")
    ax[1].set_xlim(-50, 50)
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel(r"$I/I_0$")
    ax[1].set_title("Comparación de perfiles")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ej2_red_5_rendijas.png", dpi=200)
    plt.close()


# Ejercicio 3: apertura circular y Disco de Airy

def ejercicio3():
    print("\n--- Ejercicio 3: apertura circular y Disco de Airy ---")

    lam = 632.8e-9
    R = 50e-6
    z = 1.0
    L = 3e-3
    N = 1024

    X, Y, dxp = malla(N, L)

    # Máscara circular.
    A = (X**2 + Y**2 <= R**2).astype(float)

    U_raw, x, dx = campo_fraunhofer(A, dxp, lam, z)
    I = intensidad_normalizada(U_raw)

    # Energía en la apertura y en la pantalla para verificar Parseval.
    E_ap = np.sum(np.abs(A)**2) * dxp**2
    E_pant = np.sum(np.abs(U_raw)**2 / (lam*z)**2) * dx**2
    error_parseval = np.abs(E_pant - E_ap) / E_ap

    # Radio teórico del primer anillo oscuro.
    D = 2 * R
    r_A = 1.22 * lam * z / D
    theta_min = 1.22 * lam / D

    print(f"E_ap        = {E_ap:.6e}")
    print(f"E_pantalla  = {E_pant:.6e}")
    print(f"Error Parseval = {error_parseval:.3e}")
    print(f"r_A = {r_A*1e3:.4f} mm")
    print(f"theta_min = {theta_min:.6e} rad")

    # Figura 3: escala lineal y logarítmica.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    im0 = ax[0].imshow(I, extent=[x[0]*1e3, x[-1]*1e3, x[0]*1e3, x[-1]*1e3],
                       origin="lower", cmap="inferno")
    ax[0].set_title("Disco de Airy - escala lineal")
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("y [mm]")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    I_log = np.log10(I + 1e-6)
    im1 = ax[1].imshow(I_log, extent=[x[0]*1e3, x[-1]*1e3, x[0]*1e3, x[-1]*1e3],
                       origin="lower", cmap="inferno")
    ax[1].set_title(r"Disco de Airy - $\log_{10}(I/I_0+10^{-6})$")
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel("y [mm]")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig("ej3_airy_lineal_log.png", dpi=200)
    plt.close()


# Ejercicio 4: ASM, Fresnel y Fraunhofer

def ejercicio4():
    print("\n--- Ejercicio 4: ASM, Fresnel y Fraunhofer ---")

    lam = 632.8e-9
    w = 1e-3
    L = 6e-3
    N = 1024
    z_lista = (0.05, 0.40, 5.0)

    X, Y, dxp = malla(N, L)

    # Apertura cuadrada de lado w.
    A = ((np.abs(X) <= w/2) & (np.abs(Y) <= w/2)).astype(float)

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    for j, z in enumerate(z_lista):
        U_asm, x_asm = campo_asm(A, dxp, lam, z)
        I_asm = intensidad_normalizada(U_asm)
        NF = (w/2)**2 / (lam * z)

        print(f"z = {z:.2f} m, N_F = {NF:.4f}")

        ax[0, j].imshow(I_asm, extent=[x_asm[0]*1e3, x_asm[-1]*1e3,
                                       x_asm[0]*1e3, x_asm[-1]*1e3],
                        origin="lower", cmap="inferno")
        ax[0, j].set_title(f"ASM, z={z} m")
        ax[0, j].set_xlabel("x [mm]")
        ax[0, j].set_ylabel("y [mm]")

        ax[1, j].plot(x_asm*1e3, I_asm[N//2, :])
        ax[1, j].set_xlabel("x [mm]")
        ax[1, j].set_ylabel(r"$I/I_0$")
        ax[1, j].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ej4_asm_fresnel_fraunhofer.png", dpi=200)
    plt.close()

    # Comparación en z = 5 m con Fraunhofer.
    z3 = z_lista[-1]
    U_asm, x_asm = campo_asm(A, dxp, lam, z3)
    I_asm = intensidad_normalizada(U_asm)

    U_fra, x_fra, _ = campo_fraunhofer(A, dxp, lam, z3)
    I_fra = intensidad_normalizada(U_fra)

    # Interpolación del perfil de Fraunhofer sobre la malla del ASM.
    I_fra_interp = np.interp(x_asm, x_fra, I_fra[N//2, :])

    # NRMSE sobre el perfil central.
    num = np.linalg.norm(I_asm[N//2, :] - I_fra_interp)
    den = np.linalg.norm(I_fra_interp)
    nrmse = num / den
    print(f"NRMSE en z={z3} m: {nrmse:.4f}")

    # Figura 4b: comparación ASM vs Fraunhofer.
    plt.figure(figsize=(7, 4))
    plt.plot(x_asm*1e3, I_asm[N//2, :], "b-", label="ASM")
    plt.plot(x_asm*1e3, I_fra_interp, "r--", label="Fraunhofer")
    plt.xlabel("x [mm]")
    plt.ylabel(r"$I/I_0$")
    plt.title(f"Comparación ASM vs Fraunhofer, z={z3} m")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ej4_comparacion_asm_fraunhofer.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    ejercicio1()
    ejercicio2()
    ejercicio3()
    ejercicio4()