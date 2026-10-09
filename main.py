import numpy as np
import sys

def ler_instancia(caminho_arquivo):
    num_vertices = 0
    fonte = 0
    sorvedouro = 0
    arestas = []

    with open(caminho_arquivo, "r") as f:
        for linha in f:
            partes = linha.split()
            if len(partes) == 0 or partes[0] == "c":
                continue
            
            if partes[0] == "p":
                num_vertices = int(partes[2])
            elif partes[0] == "n":
                if partes[2] == "s":
                    fonte = int(partes[1])
                elif partes[2] == "t":
                    sorvedouro = int(partes[1])
            elif partes[0] == "a":
                u = int(partes[1])
                v = int(partes[2])
                cap = int(partes[3])
                arestas.append((u, v, cap))

    return num_vertices, fonte, sorvedouro, arestas

def aplicar_pivo(tableau, linha, coluna):
    tableau[linha] = tableau[linha] / tableau[linha, coluna]
    
    for i in range(len(tableau)):
        if i != linha:
            fator = tableau[i, coluna]
            tableau[i] = tableau[i] - (fator * tableau[linha])

def rodar_simplex(tableau, base, colunas_permitidas=[]):
    for iteracao in range(10000):
        custos = tableau[-1, :-1]
        
        if len(colunas_permitidas) == 0:
            coluna = int(np.argmin(custos))
            if custos[coluna] >= -0.000001:
                break 
        else:
            pos_menor = int(np.argmin(custos[colunas_permitidas]))
            coluna = colunas_permitidas[pos_menor]
            
            if custos[coluna] >= -0.000001:
                break

        menor_razao = 999999999
        linha = -1
        
        for i in range(len(tableau) - 1):
            coeficiente = tableau[i, coluna]
            if coeficiente > 0.000001:
                razao = tableau[i, -1] / coeficiente
                if razao < menor_razao:
                    menor_razao = razao
                    linha = i
                    
        if linha == -1:
            #"O problema é ilimitado."
            return -1
            
        aplicar_pivo(tableau, linha, coluna)
        base[linha] = coluna
        
    else:
        #"estourou"
        return -1

    solucao = np.zeros(tableau.shape[1] - 1)
    for i in range(len(base)):
        col = base[i]
        solucao[col] = tableau[i, -1]
        
    return solucao

def modelar_corte_minimo(n, s, t, arestas):
    m = len(arestas)
    linhas = 2 * m + 2
    colunas = m + n + 2 * m 

    A = np.zeros((linhas, colunas))
    b = np.zeros(linhas)
    c = np.zeros(colunas)

    for k in range(len(arestas)):
        u = arestas[k][0]
        v = arestas[k][1]
        capacidade = arestas[k][2]

        col_u = m + u - 1
        col_v = m + v - 1
        ida = 2 * k
        volta = 2 * k + 1

        c[k] = capacidade 

        A[ida, k] = 1
        A[ida, col_u] = -1
        A[ida, col_v] = 1
        A[ida, m + n + ida] = -1

        A[volta, k] = 1
        A[volta, col_u] = 1
        A[volta, col_v] = -1
        A[volta, m + n + volta] = -1

    A[-2, m + s - 1] = 1
    A[-1, m + t - 1] = 1
    b[-1] = 1 

    return A, b, c

def resolver_primal_dual(A, b, c):
    linhas = len(A)
    colunas = len(A[0])
    
    tableau = np.zeros((linhas + 1, colunas + linhas + 1))
    
    for i in range(linhas):
        for j in range(colunas):
            tableau[i][j] = A[i][j]
            
    for i in range(linhas):
        tableau[i][colunas + i] = 1
        
    for i in range(linhas):
        tableau[i][-1] = b[i]
        
    for i in range(linhas):
        tableau[-1][colunas + i] = 1
        
    for i in range(linhas):
        for j in range(len(tableau[0])):
            tableau[-1][j] = tableau[-1][j] - tableau[i][j]

    base = list(range(colunas, colunas + linhas))
    artificiais = list(range(colunas, colunas + linhas))
    pi = np.zeros(linhas) 

    it = 0
    for i in range(int(100000)):
        it += 1
        custo_reduzido = c - pi.dot(A)
        #"Erro: O problema Dual perdeu a viabilidade."
        if min(custo_reduzido) < -0.000001:
            return -1

        J = []
        for j in range(colunas):
            if abs(custo_reduzido[j]) <= 0.000001:
                J.append(j)
        
        solucao = rodar_simplex(tableau, base, J + artificiais)
        
        if type(solucao) == int:
            return -1

        soma_artificiais = sum(solucao[colunas:])
        # print(f"Iteração {it}: Variáveis artificiais = {soma_artificiais:.6f} | Custo dual = {b.dot(pi):.6f}")
        
        if soma_artificiais <= 0.000001:
            solucao_final = []
            valor_otimo = 0
            
            for i in range(colunas):
                solucao_final.append(solucao[i])
                valor_otimo = valor_otimo + (c[i] * solucao[i])
                
            return solucao_final, valor_otimo, pi
            
        direcao_dual = np.zeros(linhas)
        for i in range(linhas):
            valor_artificial = tableau[-1][colunas + i]
            direcao_dual[i] = 1 - valor_artificial
            
        wa = direcao_dual.dot(A)
        
        passos = []
        for j in range(colunas):
            if j not in J:
                if wa[j] > 0.000001:
                    valor_passo = custo_reduzido[j] / wa[j]
                    passos.append(valor_passo)
        
        if len(passos) == 0:
            return -1
            
        menor_passo = min(passos)
        pi = pi + (menor_passo * direcao_dual)

if __name__ == "__main__":
    nome_arquivo = "instance3.min"
    
    n, s, t, arestas = ler_instancia(nome_arquivo)
    A, b, c = modelar_corte_minimo(n, s, t, arestas)
    
    resultado = resolver_primal_dual(A, b, c)

    if resultado == -1:
        print(f"\nA execução parou! Motivo: {resultado}")
    else:
        solucao, valor_otimo, var_duais = resultado

        num_arestas = len(arestas)
        distancias = solucao[num_arestas : num_arestas + n]
        
        conjunto_S = []
        for u in range(1, n + 1):
            if distancias[u - 1] < 0.5:
                conjunto_S.append(u)
        
        arestas_cortadas = []
        for i in range(len(arestas)):
            u = arestas[i][0]
            v = arestas[i][1]
            
            if (u in conjunto_S) != (v in conjunto_S):
                arestas_cortadas.append((u, v))
        
        print("-" * 40)
        print(f"Valor Primal (Corte Mínimo): {valor_otimo}")
        print(f"Valor Dual (Fluxo Máximo): {float(b.dot(var_duais))}")
        print(f"Vértices no lado da Fonte (S): {conjunto_S}")
        print(f"Arestas no corte: {arestas_cortadas}")