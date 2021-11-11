import numpy as np
from math import dist
import pandas as pd

df  = pd.read_csv('guia2/circulo.csv', header=None)


# Definicion
x           = df.to_numpy()


dimRed      = [5, 5]  # 10x10
etapas      = [750, 1000, 3000]
n           = 0  # Contador de etapas
epoca       = 0  # Contador de epoca
corriendo   = True
entorno     = 2


def tasa_apren(epoca, n, etp):
    if n == 0:
        return 0.9
    if n == 1:
        return (0.01 - 0.9) * epoca / etp[1]
    return 0.01


# def entorno(epoca, n, etp):  # o vecindad
#     if n == 0:
#         return 0.9
#     if n == 1:
#         return (0.01 - 0.9) * epoca / etp[1]
#     return 0.01


def neighbors(a, radius, rowNumber, columnNumber):
    return [[a[i][j] if i >= 0 and i < len(a) and j >= 0 and j < len(a[0]) else 0
             for j in range(columnNumber - 1 - radius, columnNumber + radius)]
            for i in range(rowNumber - 1 - radius, rowNumber + radius)]


while epoca < etapas[n] and corriendo:

    # Genero mis pesos
    cantEnt = len(x[0])
    w = np.random.rand(dimRed[0], dimRed[1], cantEnt)

    # Recorro cada una de mis entradas y comparo
    ganadoras = []
    for e in range(len(x)):
        # en cada x[i] tengo mi entrada (tupla) que tengo que sacar la dist menor con w[j]
        distEntradasAPesos = []
        for i in range(0, len(w)):
            for j in range(0, len(w[i])):
                distAux = dist(x[e], w[i][j])
                distEntradasAPesos.append([distAux, i, j])
        # En este punto tengo la neurona ganadora
        menorPeso = sorted(distEntradasAPesos, key=lambda l: l[0])[0]
        ganadoras.append([menorPeso[1], menorPeso[2]]) # Guardo fila y columna del menor peso

    # Entorno
    for g in range(len(ganadoras)):
        for i in range(0, len(w)):
            for j in range(0, len(w[i])):
                distAux = dist(ganadoras[g], [i, j])
                if distAux < entorno:
                    # Reajusto el peso
                    print("En la posición ", i, ",", j, " del entorno ", ganadoras[g] )

    # Recorro cada uno de mis pesos y me fijo cual es el mas cercano
    minSuma = w.max()
    minIndex = np.unravel_index(np.argmax(w, axis=None), w.shape)


    epoca = epoca + 1
    if epoca == etapas[n]:
        n = n + 1  # Voy a la siguiente etapa
        if n > 3:
            corriendo = False