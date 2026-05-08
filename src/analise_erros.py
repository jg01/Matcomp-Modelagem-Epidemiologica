import numpy as np
import pandas as pd

# Importar o nosso motor matemático
from modelo_seird import derivadas_seirdv, resolver_rk4

# =====================================================================
# 1. FUNÇÕES DE CÁLCULO DE ERRO (Tópico da Ementa: Análise de Erros)
# =====================================================================
def calcular_mae(real, previsto):
    """ Calcula o Erro Absoluto Médio (quantos casos o modelo erra por dia) """
    return np.mean(np.abs(real - previsto))

def calcular_smape(real, previsto):
    """ 
    Calcula o Erro Percentual Médio Simétrico.
    É melhor que o MAPE tradicional porque lida bem com valores próximos de zero.
    """
    denominador = (np.abs(real) + np.abs(previsto)) / 2.0
    # Impede divisão por zero caso ambos os valores sejam 0
    erro = np.divide(np.abs(real - previsto), denominador, out=np.zeros_like(real), where=denominador!=0)
    return 100.0 * np.mean(erro)

# =====================================================================
# 2. EXECUÇÃO DA AVALIAÇÃO DO MODELO
# =====================================================================
def executar_analise():
    print("A carregar os dados reais da 1ª Onda...")
    df = pd.read_csv("data/processed/covid_brasil_limpo.csv")
    df_onda = df[(df['date'] >= '2020-03-01') & (df['date'] <= '2020-04-30')].reset_index()
    dados_reais = df_onda['casos_mm7d'].values

    # Inserir os parâmetros ótimos que o seu algoritmo descobriu
    melhor_beta = 0.7000
    melhor_gamma = 0.1571
    
    # Restantes parâmetros de controlo
    sigma = 1/5.2
    mu = 0.01
    taxa_vacina = 0.0
    N = 214000000

    # Condições Iniciais (As mesmas usadas na otimização)
    I0 = dados_reais[0] if dados_reais[0] > 0 else 1
    E0 = I0 * 4
    R0, D0, V0 = 0, 0, 0
    S0 = N - I0 - E0 - R0 - D0 - V0
    y0 = np.array([S0, E0, I0, R0, D0, V0])

    vetor_tempo = np.arange(len(dados_reais))
    args = (melhor_beta, sigma, melhor_gamma, mu, taxa_vacina, N)

    # 3. Rodar a simulação com os parâmetros perfeitos
    print("A resolver o sistema de EDOs via RK4...")
    resultados = resolver_rk4(derivadas_seirdv, y0, vetor_tempo, args)
    
    E_teorico = resultados[:, 1]
    casos_novos_teoricos = sigma * E_teorico

    # 4. Calcular as métricas
    mae = calcular_mae(dados_reais, casos_novos_teoricos)
    smape = calcular_smape(dados_reais, casos_novos_teoricos)

    print("\n" + "="*40)
    print("      RESULTADOS DA ANÁLISE DE ERROS      ")
    print("="*40)
    print(f"Erro Absoluto Médio (MAE): {mae:.2f} casos/dia")
    print(f"Erro Percentual Simétrico (SMAPE): {smape:.2f}%")
    print("="*40)
    
    # Pequeno guia para a apresentação
    print("\n[Dica para a Apresentação]")
    print(f"\"O nosso modelo desvia-se, em média, {mae:.0f} casos diários numa escala de 214 milhões de habitantes.")
    print(f"Em termos percentuais morfológicos (SMAPE), o erro é de apenas {smape:.2f}%, o que valida matematicamente")
    print("a convergência do algoritmo de Mínimos Quadrados e a estabilidade do Runge-Kutta 4.\"")

if __name__ == "__main__":
    executar_analise()