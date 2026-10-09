import os
import glob

from leitor import ler_instancia
from p1_pulp import resolver_p1
from p2_pulp import resolver_p2

def main():
    pasta_instancias = "instancias"
    arquivos = sorted(glob.glob(os.path.join(pasta_instancias, "*")))

    if not arquivos:
        print(f"Nenhum arquivo encontrado na pasta '{pasta_instancias}/'.")
        return

    header = f"{'Instância':<25} | {'Problema':<8} | {'Algoritmo':<10} | {'PuLP':<10} | {'Bateu?'}"
    print(header)
    print("-" * len(header))

    for caminho_arquivo in arquivos:
        nome_instancia = os.path.basename(caminho_arquivo)

        try:
            dados = ler_instancia(caminho_arquivo)
        except Exception as err:
            print(f"{nome_instancia:<25} | Erro ao ler arquivo: {err}")
            continue

        if dados.get('tipo_problema') == 'mcf':
            problema = "P2"
            obj_pulp, tempo_pulp = resolver_p2(dados)
            
            obj_alg = None
        else:
            problema = "P1"
            obj_pulp, tempo_pulp = resolver_p1(dados)
            
            obj_alg = None

        if obj_alg is None:
            bateu = "—"
            str_alg = "—"
        else:
            str_alg = f"{obj_alg:.2f}"
            bateu = "sim" if abs(obj_alg - obj_pulp) < 1e-6 else "não"

        str_pulp = f"{obj_pulp:.2f}" if obj_pulp is not None else "—"

        print(f"{nome_instancia:<25} | {problema:<8} | {str_alg:<10} | {str_pulp:<10} | {bateu}")

if __name__ == "__main__":
    main()