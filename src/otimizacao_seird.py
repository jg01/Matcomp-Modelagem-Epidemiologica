import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importar o motor matemático manual que já validámos
from modelo_seird import derivadas_seirdv, resolver_rk4

def ajustar_curvas():
    print("A carregar os dados reais processados...")
    df = pd.read_csv("data/processed/covid_brasil_limpo.csv")
    
    # 1. Recorte temporal: Vamos treinar o modelo na 1ª Onda (Março e Abril de 2020)
    # É o período onde a dinâmica do vírus ocorreu de forma mais natural, antes das vacinas.
    df_onda = df[(df['date'] >= '2020-03-01') & (df['date'] <= '2020-04-30')].reset_index()
    dados_reais = df_onda['casos_mm7d'].values
    
    # 2. Configurar o Ambiente de Simulação
    N = 214000000
    I0 = dados_reais[0] if dados_reais[0] > 0 else 1 # Garantir pelo menos 1 infetado no dia 0
    E0 = I0 * 4         # Estimativa: por cada infetado detetado, há 4 expostos incubando
    R0, D0, V0 = 0, 0, 0
    S0 = N - I0 - E0 - R0 - D0 - V0
    y0 = np.array([S0, E0, I0, R0, D0, V0])
    
    sigma = 1/5.2 # Período de incubação de ~5 dias
    mu = 0.01     # Taxa de letalidade fixa para simplificar
    taxa_vacina = 0.0
    vetor_tempo = np.arange(len(dados_reais))
    
    # 3. Mínimos Quadrados: Busca em Grelha Manual
    print("\nA iniciar a otimização manual dos parâmetros...")
    betas = np.linspace(0.1, 0.8, 15)  # Testar 15 valores diferentes para a Transmissão
    gammas = np.linspace(0.05, 0.2, 15) # Testar 15 valores diferentes para a Recuperação
    
    melhor_erro = float('inf')
    melhor_beta = 0
    melhor_gamma = 0
    melhor_curva = None
    
    iteracoes = 0
    for b in betas:
        for g in gammas:
            args = (b, sigma, g, mu, taxa_vacina, N)
            # Rodar o RK4 para esta combinação de parâmetros
            resultados = resolver_rk4(derivadas_seirdv, y0, vetor_tempo, args)
            
            # Extrair os Casos Novos Teóricos (fluxo de Expostos para Infetados)
            E_teorico = resultados[:, 1]
            casos_novos_teoricos = sigma * E_teorico
            
            # Calcular o Erro Quadrático (MSE)
            erro = np.mean((casos_novos_teoricos - dados_reais)**2)
            
            if erro < melhor_erro:
                melhor_erro = erro
                melhor_beta = b
                melhor_gamma = g
                melhor_curva = casos_novos_teoricos
            
            iteracoes += 1
            
    print(f"Otimização concluída após {iteracoes} simulações de RK4.")
    print(f"--- PARÂMETROS IDEAIS ENCONTRADOS ---")
    print(f"Taxa de Transmissão (Beta): {melhor_beta:.4f}")
    print(f"Taxa de Recuperação (Gamma): {melhor_gamma:.4f}")
    
    # 4. Gerar o gráfico do "Casamento"
    plt.figure(figsize=(12, 6))
    plt.plot(df_onda['date'], dados_reais, 'ro-', alpha=0.6, label='Realidade (Média Móvel)')
    plt.plot(df_onda['date'], melhor_curva, 'b--', linewidth=3, 
            label=f'Previsão RK4 Ajustada (Beta={melhor_beta:.2f})')
    
    plt.title('Calibração do Modelo Matemático (Mínimos Quadrados) - 1ª Onda no Brasil', fontsize=14)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Novos Casos Diários', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('data/processed/casamento_curvas.png', dpi=300)
    print("Gráfico guardado em: data/processed/casamento_curvas.png")
    plt.show()

if __name__ == "__main__":
    ajustar_curvas()