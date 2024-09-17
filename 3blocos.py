# Programa desenvolvido para a disciplina de Tópicos em Controle Avançado
# Prof.: Dr. Vitor Gervini (vitor.gervini@gmail.com)
# Fundação Universidade Federal do Rio Grande - FURG
# Data: 20/08/2020

import numpy as np
import matplotlib.pyplot as plt
from vpython import *


def dados_simul():
    RK4 = lambda f: lambda x, u, dt: (lambda dx1: (
        lambda dx2: (lambda dx3: (lambda dx4: (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6)(dt * f(x + dx3, u)))(
            dt * f(x + dx2 / 2, u)))(dt * f(x + dx1 / 2, u)))(dt * f(x, u))
    dx = RK4(lambda x, u: A @ x + B @ u)
    dx_est = RK4(lambda x_est, u: A @ x_est + B @ u + L @ (y - y_est))

    m1 = 2.0;
    m2 = 1.0;
    m3 = 1.5;
    k1 = 10.0;
    k2 = 10.0;
    k3 = 15.0;
    b1 = 0;
    b2 = 1.0;
    b3 = 0.5
    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [-(k1 + k2) / m1, k2 / m1, 0, -(b1 + b2) / m1, b2 / m1, 0],
                  [k2 / m2, -(k2 + k3) / m2, k3 / m2, b2 / m2, -(b2 + b3) / m2, b3 / m2],
                  [0, k3 / m3, -k3 / m3, 0, b3 / m3, -b3 / m3]])
    B = np.array([[0], [0], [0], [1 / m1], [0], [0]])
    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])
    K = np.array([[100, 50, 25, 20, 10, 5]])
    N = np.array([[50]])
    L = np.array([[100, 0, 0],
                  [0, 100, 0],
                  [0, 0, 100],
                  [200, 0, 0],
                  [0, 200, 0],
                  [0, 0, 200]])

    t, tf, dt, u, x, r = 0, 10, .01, np.array([[0]]), np.array([[0.5], [1], [1.5], [0], [0], [0]]), np.array([[.5]])
    x_est = np.array([[0], [0], [0], [0], [0], [0]])
    X, U, T = x, u, t

    for i in range(int((tf - t) / dt)):
        u = (N @ r - K @ x_est)
        t, x = t + dt, x + dx(x, u, dt)
        y, y_est = C @ x, C @ x_est
        x_est = x_est + dx_est(x_est, u, dt)  # estimação do estado
        X, U, T = np.append(X, x, axis=1), np.append(U, u, axis=1), np.append(T, t)

    f1, f2, f3 = U[0], U[0] * 0, U[0] * 0

    return T, X, f1, f2, f3


def imprime(T, X, f1, f2, f3):
    # Gráfico de Posição
    plt.plot(T, X[0], 'k', label='x1')
    plt.plot(T, X[1], 'b', label='x2')
    plt.plot(T, X[2], 'r', label='x3')
    plt.xlabel('tempo (s)')
    plt.ylabel('Posição (m)')
    plt.title('Gráfico de Posição')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close('all')

    # Gráfico de Força
    plt.figure()
    plt.plot(T, f1, 'k', label='f1')
    plt.plot(T, f2, 'b', label='f2')
    plt.plot(T, f3, 'r', label='f3')
    plt.xlabel('tempo (s)')
    plt.ylabel('Força (N)')
    plt.title('Gráfico de Força')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close('all')

    # Gráfico de Velocidade
    plt.figure()
    plt.plot(T, X[3], 'k', label='v1')
    plt.plot(T, X[4], 'b', label='v2')
    plt.plot(T, X[5], 'r', label='v3')
    plt.xlabel('tempo (s)')
    plt.ylabel('Velocidade (m/s)')
    plt.title('Gráfico de Velocidade')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close('all')

    # Gráfico de Aceleração
    v1, v2, v3 = X[3], X[4], X[5]
    a1 = np.gradient(v1, T)
    a2 = np.gradient(v2, T)
    a3 = np.gradient(v3, T)

    plt.figure()
    plt.plot(T, a1, 'k', label='a1')
    plt.plot(T, a2, 'b', label='a2')
    plt.plot(T, a3, 'r', label='a3')
    plt.xlabel('tempo (s)')
    plt.ylabel('Aceleração (m/s²)')
    plt.title('Gráfico de Aceleração')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close('all')



def inicializa(T, X):
    tam_eixo, tam_cubo, tam_mola, esp_chao = 2, 2, 5, .05
    cena1 = canvas(title='Simulação massa-mola-amortecedor com controle', width=640, height=300, center=vector(8, 0, 0),
                   background=color.white)
    dir1 = vector(1, 0, 0)
    forca1 = arrow(pos=vector(0, tam_cubo, 0), axis=dir1, color=color.green)
    forca2 = arrow(pos=vector(0, tam_cubo, 0), axis=dir1, color=color.red)
    forca3 = arrow(pos=vector(0, tam_cubo, 0), axis=dir1, color=color.blue)
    mola1 = helix(axis=dir1, thickness=.2, color=color.blue)
    mola2 = helix(axis=dir1, thickness=.2, color=color.blue)
    mola3 = helix(axis=dir1, thickness=.2, color=color.blue)
    arrow(axis=vector(tam_eixo, 0, 0), color=color.red), arrow(axis=vector(0, tam_eixo, 0), color=color.green), arrow(
        axis=vector(0, 0, tam_eixo), color=color.blue)
    massa1 = box(opacity=.5, size=2 * tam_cubo * vec(1, 1, 1), color=color.green)
    massa2 = box(opacity=.5, size=2 * tam_cubo * vec(1, 1, 1), color=color.red)
    massa3 = box(opacity=.5, size=2 * tam_cubo * vec(1, 1, 1), color=color.blue)
    chao = box(pos=vec(15, -(tam_cubo + esp_chao), 0), size=vec(30, 2 * esp_chao, 2 * tam_cubo), color=vec(.8, .8, .8))
    graf1 = graph(title='Posição', width=600, height=300, xtitle='<i>t</i> (s)',
                  ytitle='<i>x</i><sub>1</sub> (m)    <i>x</i><sub>2</sub> (m)   <i>x</i><sub>3</sub> (m)',
                  fast=True, xmin=T.min(), xmax=T.max())
    graf2 = graph(title='Força', width=600, height=300, xtitle='<i>t</i> (s)',
                  ytitle='<i>F</i><sub>1</sub> (N)  <i>F</i><sub>2</sub> (N)  <i>F</i><sub>3</sub> (N)',
                  fast=True, xmin=T.min(), xmax=T.max())
    graf3 = graph(title='Velocidade', width=600, height=300, xtitle='<i>t</i> (s)',
                  ytitle='<i>v</i><sub>1</sub> (m/s)    <i>v</i><sub>2</sub> (m/s)    <i>v</i><sub>3</sub> (m/s)',
                  fast=True, xmin=T.min(), xmax=T.max())
    return (forca1, forca2, forca3, mola1, mola2, mola3, massa1, massa2, massa3,
            gcurve(graph=graf1, color=color.green), gcurve(graph=graf1, color=color.red),
            gcurve(graph=graf1, color=color.blue),
            gcurve(graph=graf2, color=color.green), gcurve(graph=graf2, color=color.red),
            gcurve(graph=graf2, color=color.blue),
            gcurve(graph=graf3, color=color.green), gcurve(graph=graf3, color=color.red),
            gcurve(graph=graf3, color=color.blue),
            tam_cubo, tam_mola)


def move(T, X, f1, f2, f3, tam_cubo, tam_mola, forca1, forca2, forca3, mola1, mola2, mola3, massa1, massa2, massa3, gx1,
         gx2, gx3, gf1, gf2, gf3, gv1, gv2, gv3):
    delta_f = lambda x: 0 if x < 0 else 2 * tam_cubo

    # Gráficos extras
    graf_aceleracao = graph(title='Aceleração', width=600, height=300, xtitle='<i>t</i> (s)',
                            ytitle='<i>a</i> (m/s²)', fast=True, xmin=T.min(), xmax=T.max())



    ga1 = gcurve(graph=graf_aceleracao, color=color.green)
    ga2 = gcurve(graph=graf_aceleracao, color=color.red)
    ga3 = gcurve(graph=graf_aceleracao, color=color.blue)

    disp_rate = 1 / (T[1] - T[0])
    x1, x2, x3, v1, v2, v3 = X[0] + tam_mola, X[1] + 2 * (tam_mola + tam_cubo), X[2] + 3 * (tam_mola + 2 * tam_cubo), X[
        3], X[4], X[5]

    # Calculando as acelerações
    a1 = np.gradient(v1, T)
    a2 = np.gradient(v2, T)
    a3 = np.gradient(v3, T)

    for i in range(len(T)):
        rate(disp_rate)
        mola1.axis.x = x1[i]
        massa1.pos.x = x1[i] + tam_cubo
        mola2.pos.x, mola2.axis.x = x1[i] + 2 * tam_cubo, x2[i] - x1[i] - 2 * tam_cubo
        massa2.pos.x = x2[i] + tam_cubo
        mola3.pos.x, mola3.axis.x = x2[i] + 2 * tam_cubo, x3[i] - x2[i] - 2 * tam_cubo
        massa3.pos.x = x3[i] + tam_cubo
        forca1.pos.x, forca1.axis.x = x1[i] + delta_f(f1[i]), f1[i] / 2
        forca2.pos.x, forca2.axis.x = x2[i] + delta_f(f2[i]), f2[i] / 2
        forca3.pos.x, forca3.axis.x = x3[i] + delta_f(f3[i]), f3[i] / 2

        # Plotagem das posições
        gx1.plot(T[i], X[0][i]), gx2.plot(T[i], X[1][i]), gx3.plot(T[i], X[2][i])

        # Plotagem das velocidades
        gv1.plot(T[i], X[3][i]), gv2.plot(T[i], X[4][i]), gv3.plot(T[i], X[5][i])

        # Plotagem das forças
        gf1.plot(T[i], f1[i]), gf2.plot(T[i], f2[i]), gf3.plot(T[i], f3[i])

        # Plotagem das acelerações
        ga1.plot(T[i], a1[i]), ga2.plot(T[i], a2[i]), ga3.plot(T[i], a3[i])


T, X, f1, f2, f3 = dados_simul()
imprime(T, X, f1, f2, f3)
forca1, forca2, forca3, mola1, mola2, mola3, massa1, massa2, massa3, gx1, gx2, gx3, gf1, gf2, gf3, gv1, gv2, gv3, tam_cubo, tam_mola = inicializa(
    T, X)
from time import sleep

sleep(3)
move(T, X, f1, f2, f3, tam_cubo, tam_mola, forca1, forca2, forca3, mola1, mola2, mola3, massa1, massa2, massa3, gx1,
     gx2, gx3, gf1, gf2, gf3, gv1, gv2, gv3)



# Mantém a janela aberta
while True:
    rate(1)  # Define uma taxa de atualização lenta para manter a janela aberta
