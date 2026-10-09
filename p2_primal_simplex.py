# pip install numpy
# pip install pytest

# Coisas que percebi enquanto fazia o trabalho:

# 1) O PDF não pede conservação de fluxo no destino, então o fluxo pode entrar e sair dele. Fiz do jeito que o enunciado pede.
# 2) A capacidade das arestas eu tratei como "por sentido", igual está no PDF. Se fosse a soma dos dois sentidos, ia dar diferente.

import numpy as np
import unittest


class SimplexSolver:
    """

    """

    def __init__(self, c, A, b, maximise=False):
        """
        c: coeficientes da funcao objetivo (tamanho n)
        A: matriz de restricoes (m x n) -- restricoes de <=
        b: lado direito (tamanho m) -- deve ser >= 0
        maximise: True para maximizacao, False para minimizacao
        """
        if maximise:
            c = [-ci for ci in c]  # converting max para min

        self.is_maximise = maximise
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.m, self.n = self.A.shape
        self.tableau = None
        self.basic_vars = []  # indices das variaveis basicas
        self.history = []  # para visualizacao do caminho

    def initialize_tableau(self):
        """
        Constroi o tableau inicial do Simplex.
        Layout:
            [ A | I | b ]
            [ -c | 0 | 0 ]
        """
        # Adiciona variaveis de folga (identidade m x m)
        folgas = np.eye(self.m)

        # Matriz completa: [A | I | b]
        self.tableau = np.hstack([self.A, folgas,
                                  self.b.reshape(-1, 1)])

        # Linha da FO: [c | 0 | 0]
        fo_row = np.hstack([self.c, np.zeros(self.m + 1)])

        # Empilha a linha da FO abaixo
        self.tableau = np.vstack([self.tableau, fo_row])

        # Variaveis basicas iniciais: indices n a n+m-1 (folgas)
        self.basic_vars = list(range(self.n, self.n + self.m))

        print("Tableau inicial:")
        print(self.tableau)
        print(f"Variaveis basicas: {self.basic_vars}")

    def find_entering_var(self, tol=1e-10):
        """
        Encontra o indice da variavel que entra na base.
        Retorna None se o tableau ja e otimo.
        """
        # Ultima linha (FO), excluindo o RHS
        fo_row = self.tableau[-1, :-1]

        # Se todos os custos reduzidos >= -tol, e otimo
        if np.all(fo_row >= -tol):
            return None

        # Regra de Dantzig: menor custo reduzido
        entering = np.argmin(fo_row)

        print(f"Variavel entrando: x_{entering} "
              f"(custo reduzido = {fo_row[entering]:.4f})")
        return entering

    def find_leaving_var(self, entering_col, tol=1e-10):
        """
        Encontra a linha do pivo usando o teste da razao minima.
        Retorna None se o problema e ilimitado.
        """
        col = self.tableau[:-1, entering_col]  # sem a linha FO
        rhs = self.tableau[:-1, -1]  # RHS

        # Inicializa razoes como infinito
        ratios = np.full_like(rhs, np.inf)

        # Calcula razao apenas onde col > tol
        positive_mask = col > tol
        ratios[positive_mask] = rhs[positive_mask] / \
                                col[positive_mask]

        # Se todas razoes sao infinito, problema e ilimitado
        if np.all(np.isinf(ratios)):
            print("PROBLEMA ILIMITADO!")
            return None

        # Linha do pivo: menor razao
        leaving = np.argmin(ratios)

        print(f"Variavel saindo: indice {leaving} "
              f"(razao = {ratios[leaving]:.4f})")
        print(f"Razoes: {ratios}")
        return leaving

    def pivot(self, row, col):
        """
        Realiza o pivoteamento no elemento (row, col).
        """
        # 1. Normaliza a linha do pivo
        self.tableau[row] /= self.tableau[row, col]

        # 2. Zera as outras entradas da coluna do pivo
        for i in range(self.tableau.shape[0]):
            if i != row:
                fator = self.tableau[i, col]
                self.tableau[i] -= fator * self.tableau[row]

        # 3. Atualiza a lista de variaveis basicas
        self.basic_vars[row] = col

        print(f"Pivoteamento em (linha {row}, coluna {col})")
        print("Novo tableau:")
        print(self.tableau)
        print(f"Variaveis basicas: {self.basic_vars}")

    def solve(self, max_iter=1000, verbose=True):
        """
        Executa o algoritmo Simplex completo.
        Retorna (x_opt, z_opt) ou None se ilimitado.
        """
        self.initialize_tableau()

        for iteration in range(max_iter):
            if verbose:
                print(f"\n--- Iteracao {iteration} ---")

            # Encontrar variavel entrante
            entering = self.find_entering_var()
            if entering is None:
                if verbose:
                    print("OTIMO ENCONTRADO!")
                break

            # Encontrar variavel saindo
            leaving = self.find_leaving_var(entering)
            if leaving is None:
                print("PROBLEMA ILIMITADO")
                return None

            # Pivoteamento
            self.pivot(leaving, entering)

            # Salvar estado para visualizacao
            x, _ = self.get_solution()
            self.history.append(x.copy())

        # Extrair solucao final
        x_opt, z_opt = self.get_solution()
        return x_opt, z_opt

    def get_solution(self):
        """
        Extrai a solucao otima do tableau.
        Retorna (x, z) onde x e o vetor de variaveis originais
        e z e o valor otimo da FO.
        """
        n_total = self.n + self.m
        x = np.zeros(n_total)

        # Preenche valores das variaveis basicas
        for i, bv in enumerate(self.basic_vars):
            x[bv] = self.tableau[i, -1]

        # Valor otimo (canto inferior direito)
        # Para minimizacao, z_opt = -tableau[-1, -1]
        z_opt = self.tableau[-1, -1] if self.is_maximise else -self.tableau[-1, -1]

        # Retorna apenas as variaveis originais (sem folgas)
        return x[:self.n], z_opt
    
def montarProblema(V, E, k, arestas, commodits):
    col = {}
    contador = 0

    # Lê as commodits
    for i in range(1, k + 1):      
        for(u, v, cap) in arestas:
            col[(i, u, v)] = contador
            contador += 1

    c = [0.0] * (k * E)

    for i in range(1, k + 1):
        s, t, p = commodits[i - 1]
        for (u, v, cap) in arestas:
            if v == t:
                c[col[(i, u, v)]] = p
    
    A = []
    b = []

    # soma dos fluxos na aresta <= cap
    for (u, v, cap) in arestas:
        linha = [0.0] * (k * E)
        for i in range(1, k + 1):
            linha[col[(i, u, v)]] = 1.0
        
        A.append(linha)
        b.append(cap)
    
    # Cada igualdade vira duas desigualdades
    for i in range(1, k + 1):
        s, t, p = commodits[i - 1]
        for u in range(1, V + 1):       
            if u == s or u == t:
                continue                # pula inicio e fim

            linha = [0.0] * (k * E)
            for (a, w, cap) in arestas:
                if a == u:              # aresta sai de u
                    linha[col[(i, a, w)]] += 1.0
                if w == u:             # aresta entra em u
                    linha[col[(i, a, w)]] -= 1.0

            A.append(linha)
            b.append(0.0)

            # Linha oposta (multiplica por -1)
            A.append([-x for x in linha])
            b.append(0.0)
    
    return c, A, b, col

def lerEntradas(caminho):
    arestas = []
    inicios = {}
    fins = {}
    pesos = {}


    arquivo = open(caminho, "r")
    for linha in arquivo:
        if linha.startswith("c"):  # Ignora os comentários
            continue

        elif linha.startswith("p"):
            parte = linha.split()
            V = int(parte[2])
            E = int(parte[3])
            k = int(parte[4])

        elif linha.startswith("n"):
            parte = linha.split()
            no = int(parte[1])
            tipo = parte[2]
            commodit = int(parte[3])
            if tipo == 's':
                inicios[commodit] = no
            else:
                fins[commodit] = no

        elif linha.startswith("w"):
            parte = linha.split()
            commodit = int(parte[1])
            peso = float(parte[2])
            pesos[commodit] = peso

        elif linha.startswith("a"):    
            parte = linha.split()
            u = int(parte[1])
            v = int(parte[2])
            cap = float(parte[3])
            arestas.append((u, v, cap))
    
    # Junta o inicio, o fim e o peso de cada commodit
    commodits = []
    for i in range(1, k + 1):
        commodits.append((inicios[i], fins[i], pesos[i]))

    arquivo.close()
    return V, E, k, arestas, commodits

if __name__ == "__main__":

    V, E, k, arestas, commodits = lerEntradas("mc_instance5.max")
    print("V =", V)
    print("E =", E)
    print("k =", k)
    print("arestas =", arestas)
    print("commodits =", commodits)

    c, A, b, col = montarProblema(V, E, k, arestas, commodits)
    print()
    print("len(c) =", len(c))
    print("len(A) =", len(A))
    print("len(b) =", len(b))
    print("c =", c)
    print("b =", b)

    print()
    print("=== Resolvendo com Simplex ===")
    solver = SimplexSolver(c, A, b, maximise=True)
    resultado = solver.solve(verbose=False)
    
    # Mostra o resultado

    if resultado is None:
        print("PROBLEMA ILIMITADO")
    else:
        x, z = resultado
        print(f"Valor ótimo: {z:.4f}")
        print()
        print("Fluxos não-nulos:")
        for i in range(1, k + 1):
            s, t, p = commodits[i - 1]
            print(f"  Commodity {i} (s={s}, t={t}, p={p}):")
            for (u, v, cap) in arestas:
                valor = x[col[(i, u, v)]]
                if valor > 1e-6:
                    print(f"    f_{i}_{u}_{v} = {valor:.4f}")
