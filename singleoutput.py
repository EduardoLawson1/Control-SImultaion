import numpy as np

# Definindo as matrizes A e C
A = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],
              [-(10+10)/2.0, 10/2.0, 0, -(0+1.0)/2.0, 1.0/2.0, 0],
              [10/1.0, -(10+15)/1.0, 15/1.0, 1.0/1.0, -(1.0+0.5)/1.0, 0.5/1.0],
              [0, 15/1.5, -15/1.5, 0, 0.5/1.5, -0.5/1.5]])

C = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0]])

# Função para gerar a matriz de observabilidade
def observability_matrix(A, C):
    n = A.shape[0]
    O = C
    for i in range(1, n):
        O = np.vstack((O, np.dot(C, np.linalg.matrix_power(A, i))))
    return O

# Calculando a matriz de observabilidade
O = observability_matrix(A, C)

# Calculando o posto da matriz de observabilidade
rank_O = np.linalg.matrix_rank(O)

# Definindo a precisão para 4 casas decimais
# Definindo a precisão para 4 casas decimais
np.set_printoptions(precision=4, suppress=True)

#Printando
print("Matriz de observabilidade:\n", O)
print("Posto da matriz de observabilidade:", rank_O)

# Verificando se o sistema é observável
if rank_O == A.shape[0]:
    print("O sistema é observável.")
else:
    print("O sistema não é observável.")
