import numpy as np

m1 = 2.0;
m2 = 1.0;
m3 = 1.5;
k1 = 10.0;
k2 = 10.0;
k3 = 15.0;
b1 = 0;
b2 = 1.0;
b3 = 0.5
# Definição das matrizes A, B e C
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

# Calculando as potências de A
A_powers = [np.linalg.matrix_power(A, i) for i in range(6)]

# Calculando a matriz de controlabilidade
controllability_matrix = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(6)])

# Calculando a matriz de observabilidade
observability_matrix = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(6)])

print("Matriz de Controlabilidade:")
print(controllability_matrix)

print("\nMatriz de Observabilidade:")
print(observability_matrix)
