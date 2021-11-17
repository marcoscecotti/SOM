import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def SOM(X,cantFilas,cantColumnas):
    cantEntradas = len(X)

    #1- Inicialización: Elijo aleatoriamente las neuronas
    cantNeuronas = cantFilas * cantColumnas
    indicesNeuronas = np.random.randint(0,cantEntradas,cantNeuronas)
    neuronas = np.zeros((cantFilas,cantColumnas,len(X[0])))
    k = 0
    for i in range(cantFilas):
        for j in range(cantColumnas):
            neuronas[i][j][:] = X[indicesNeuronas[k]]
            k+=1

    neuronasIniciales = np.copy(neuronas)

    it = 0
    maxIt = 10
    entorno = 3
    tasaAprendizaje = 0.9
    while it < maxIt:
        #Para cada patron:
        for i in range(cantEntradas):
            # 2- Selección del ganador:
            # Calculo la neurona con pesos mas parecidos al patron de entrada
            distancias = np.zeros((cantFilas,cantColumnas))
            for j in range(cantFilas):
                for k in range(cantColumnas):
                    distancias[j][k] = np.linalg.norm(X[i][:] - neuronas[j][k][:])
            neuronaMin = np.unravel_index(distancias.argmin(),distancias.shape) #Pos. de la neurona

            #3- Defino qué neuronas se van a actualizar según la vecindad:
            vecinas=[]
            contador = 0
            for j in range(cantFilas):
                for k in range(cantColumnas):
                    distancia = abs(neuronaMin[0]-j) + abs(neuronaMin[1]-k)
                    if distancia <= entorno:
                        vecinas.insert(contador,(j,k))

            #4- Adaptación de los pesos para la neurona ganadora y sus vecinas
            for j in range(len(vecinas)):
                posicion = vecinas[j]
                neuronas[posicion][:] = neuronas[posicion][:] + tasaAprendizaje*(X[i][:]-neuronas[posicion][:])
        it+=1

    def tasa_apren(epoca):
        return (-0.8/200)*epoca + 0.9

    def neighbours(epoca):
        return (-2/200)*epoca + 3
    it = 0
    maxIt = 20
    while it < maxIt:
        # Para cada patron:
        for i in range(cantEntradas):
            # 2- Selección del ganador:
            # Calculo la neurona con pesos mas parecidos al patron de entrada
            distancias = np.zeros((cantFilas, cantColumnas))
            for j in range(cantFilas):
                for k in range(cantColumnas):
                    distancias[j][k] = np.linalg.norm(X[i][:] - neuronas[j][k][:])
            neuronaMin = np.unravel_index(distancias.argmin(), distancias.shape)  # Pos. de la neurona

            # 3- Defino qué neuronas se van a actualizar según la vecindad:
            vecinas = []
            contador = 0
            for j in range(cantFilas):
                for k in range(cantColumnas):
                    distancia = abs(neuronaMin[0] - j) + abs(neuronaMin[1] - k)
                    if distancia <= neighbours(it):
                        vecinas.insert(contador,(j,k))

            # 4- Adaptación de los pesos para la neurona ganadora y sus vecinas
            for j in range(len(vecinas)):
                posicion = vecinas[j]
                neuronas[posicion][:] = neuronas[posicion][:] + tasa_apren(it) * (X[i][:] - neuronas[posicion][:])

        it+=1

    it = 0
    maxIt = 60
    entorno = 0
    tasaAprendizaje = 0.1
    while it < maxIt:
        # Para cada patron:
        for i in range(cantEntradas):
            # 2- Selección del ganador:
            # Calculo la neurona con pesos mas parecidos al patron de entrada
            distancias = np.zeros((cantFilas, cantColumnas))
            for j in range(cantFilas):
                for k in range(cantColumnas):
                    distancias[j][k] = np.linalg.norm(X[i][:] - neuronas[j][k][:])
            neuronaMin = np.unravel_index(distancias.argmin(), distancias.shape)  # Pos. de la neurona

            # 3- Defino qué neuronas se van a actualizar según la vecindad:
            vecinas = []
            contador = 0
            for j in range(cantFilas):
                for k in range(cantColumnas):
                    distancia = abs(neuronaMin[0] - j) + abs(neuronaMin[1] - k)
                    if distancia <= entorno:
                        vecinas.insert(contador,(j,k))

            # 4- Adaptación de los pesos para la neurona ganadora y sus vecinas
            for j in range(len(vecinas)):
                posicion = vecinas[j]
                neuronas[posicion][:] = neuronas[posicion][:] + tasaAprendizaje * (X[i][:] - neuronas[posicion][:])
        it += 1

    neuronasFinales = np.copy(neuronas)
    return neuronasIniciales,neuronasFinales

XTrain = pd.read_csv('icgtp2datos/te.csv', header=None).to_numpy()
cantFilas = 10
cantColumnas = 10
neuronasI,neuronasF = SOM(XTrain,cantFilas,cantColumnas)
cantEntradas = len(XTrain)

plt.figure(1)
for i in range(cantEntradas):
    plt.plot(XTrain[i][0], XTrain[i][1], marker='.', color='black')

for i in range(cantFilas):
    for j in range(cantColumnas):
        plt.plot(neuronasI[i][j][0], neuronasI[i][j][1], marker='o', color='blue')

for i in range(cantFilas):
    for j in range(cantColumnas):
        if i - 1 >= 0:
            x_values = [neuronasI[i][j][0], neuronasI[i - 1][j][0]]
            y_values = [neuronasI[i][j][1], neuronasI[i - 1][j][1]]
            plt.plot(x_values, y_values)
        if j - 1 >= 0:
            x_values = [neuronasI[i][j][0], neuronasI[i][j - 1][0]]
            y_values = [neuronasI[i][j][1], neuronasI[i][j - 1][1]]
            plt.plot(x_values, y_values)
        if i + 1 <= cantFilas - 1:
            x_values = [neuronasI[i][j][0], neuronasI[i + 1][j][0]]
            y_values = [neuronasI[i][j][1], neuronasI[i + 1][j][1]]
            plt.plot(x_values, y_values)
        if j + 1 <= cantColumnas - 1:
            x_values = [neuronasI[i][j][0], neuronasI[i][j + 1][0]]
            y_values = [neuronasI[i][j][1], neuronasI[i][j + 1][1]]
            plt.plot(x_values, y_values)

plt.figure(2)
for i in range(cantEntradas):
    plt.plot(XTrain[i][0], XTrain[i][1], marker='.', color='black')

for i in range(cantFilas):
    for j in range(cantColumnas):
        plt.plot(neuronasF[i][j][0], neuronasF[i][j][1], marker='+', color='red')

for i in range(cantFilas):
    for j in range(cantColumnas):
        if i - 1 >= 0:
            x_values = [neuronasF[i][j][0], neuronasF[i - 1][j][0]]
            y_values = [neuronasF[i][j][1], neuronasF[i - 1][j][1]]
            plt.plot(x_values, y_values)
        if j - 1 >= 0:
            x_values = [neuronasF[i][j][0], neuronasF[i][j - 1][0]]
            y_values = [neuronasF[i][j][1], neuronasF[i][j - 1][1]]
            plt.plot(x_values, y_values)
        if i + 1 <= cantFilas - 1:
            x_values = [neuronasF[i][j][0], neuronasF[i + 1][j][0]]
            y_values = [neuronasF[i][j][1], neuronasF[i + 1][j][1]]
            plt.plot(x_values, y_values)
        if j + 1 <= cantColumnas - 1:
            x_values = [neuronasF[i][j][0], neuronasF[i][j + 1][0]]
            y_values = [neuronasF[i][j][1], neuronasF[i][j + 1][1]]
            plt.plot(x_values, y_values)

plt.show()
