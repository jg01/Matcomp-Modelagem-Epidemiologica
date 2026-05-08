import numpy as np

# =====================================================================
# 1. DEFINIÇÃO DO SISTEMA DE EQUAÇÕES (SEIRD-V)
# =====================================================================
def derivadas_seirdv(t, y, beta, sigma, gamma, mu, taxa_vacina, N):
    """
    Sistema de Equações Diferenciais Ordinárias (EDOs) do modelo SEIRD-V.
    """
    S, E, I, R, D, V = y
    
    # As equações de fluxo entre os compartimentos
    dSdt = -(beta * S * I) / N - (taxa_vacina * S)
    dEdt = (beta * S * I) / N - (sigma * E)
    dIdt = (sigma * E) - (gamma * I) - (mu * I)
    dRdt = (gamma * I)
    dDdt = (mu * I)
    dVdt = (taxa_vacina * S)
    
    return np.array([dSdt, dEdt, dIdt, dRdt, dDdt, dVdt])


# =====================================================================
# 2. NÚCLEO ALGÓRITMICO: RUNGE-KUTTA 4ª ORDEM (RK4) MANUAL
# =====================================================================
def rk4_passo(f, t, y, h, args):
    """
    Calcula um único passo do método de Runge-Kutta de 4ª Ordem.
    """
    k1 = h * f(t, y, *args)
    k2 = h * f(t + h/2, y + k1/2, *args)
    k3 = h * f(t + h/2, y + k2/2, *args)
    k4 = h * f(t + h, y + k3, *args)
    
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def resolver_rk4(f, y0, t, args):
    """
    Resolve o sistema de EDOs iterando o RK4 ao longo do vetor de tempo.
    """
    n_passos = len(t)
    n_variaveis = len(y0)
    
    # Matriz para guardar os resultados (cada linha é um dia, cada coluna é um compartimento S,E,I,R,D,V)
    y_resultado = np.zeros((n_passos, n_variaveis))
    y_resultado[0] = y0
    
    # Loop de integração numérica diária
    for i in range(n_passos - 1):
        h = t[i+1] - t[i] # Tamanho do passo (normalmente 1 dia)
        y_resultado[i+1] = rk4_passo(f, t[i], y_resultado[i], h, args)
        
    return y_resultado


# =====================================================================
# 3. TESTE DO MOTOR MATEMÁTICO (Simulação Inicial)
# =====================================================================
if __name__ == "__main__":
    # População aproximada do Brasil
    N = 214000000 
    
    # Condições Iniciais [S, E, I, R, D, V] no Dia 0
    I0 = 100          # 100 infetados iniciais
    E0 = 400          # Expostos (ainda não transmitem)
    R0, D0, V0 = 0, 0, 0
    S0 = N - I0 - E0 - R0 - D0 - V0
    y0 = np.array([S0, E0, I0, R0, D0, V0])
    
    # Parâmetros epidemiológicos fictícios (apenas para teste do RK4)
    beta = 0.35       # Taxa de transmissão
    sigma = 1/5.2     # Taxa de incubação (1 / dias de incubação)
    gamma = 1/14.0    # Taxa de recuperação (1 / dias infetado)
    mu = 0.01         # Taxa de mortalidade
    taxa_vacina = 0.0 # Sem vacina no dia 0
    
    argumentos_modelo = (beta, sigma, gamma, mu, taxa_vacina, N)
    
    # Simular durante 100 dias
    vetor_tempo = np.arange(0, 100, 1) # [0, 1, 2, ..., 99]
    
    print("A iniciar integração numérica (RK4)...")
    resultados = resolver_rk4(derivadas_seirdv, y0, vetor_tempo, argumentos_modelo)
    
    print(f"Fim da simulação. Infetados no dia 100: {int(resultados[-1, 2])}")