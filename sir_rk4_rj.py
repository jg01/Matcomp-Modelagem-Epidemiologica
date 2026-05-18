"""
Projeto COC351 - Modelo SIR com RK4 para COVID-19 no Rio de Janeiro

Este script:
1. Lê o banco tratado RJ_COVID_tratado.csv;
2. Prepara as variáveis epidemiológicas;
3. Implementa o modelo SIR;
4. Resolve o sistema de EDOs com Runge-Kutta de 4ª ordem (RK4);
5. Ajusta beta e gamma por mínimos quadrados;
6. Gera gráficos comparando dados reais e simulados.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ============================================================
# 1. CONFIGURAÇÕES GERAIS
# ============================================================

DATA_PATH = "RJ_COVID_tratado.csv"
OUTPUT_DIR = "graficos"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. MODELO SIR
# ============================================================

def sir_derivatives(S, I, R, beta, gamma, N):
    """
    Calcula as derivadas do modelo SIR clássico.

    dS/dt = -beta*S*I/N
    dI/dt = beta*S*I/N - gamma*I
    dR/dt = gamma*I
    """

    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I

    return dSdt, dIdt, dRdt


def rk4_step(S, I, R, beta, gamma, N, h=1.0):
    """
    Executa um passo do método de Runge-Kutta de 4ª ordem.
    """

    k1S, k1I, k1R = sir_derivatives(S, I, R, beta, gamma, N)

    k2S, k2I, k2R = sir_derivatives(
        S + 0.5 * h * k1S,
        I + 0.5 * h * k1I,
        R + 0.5 * h * k1R,
        beta,
        gamma,
        N
    )

    k3S, k3I, k3R = sir_derivatives(
        S + 0.5 * h * k2S,
        I + 0.5 * h * k2I,
        R + 0.5 * h * k2R,
        beta,
        gamma,
        N
    )

    k4S, k4I, k4R = sir_derivatives(
        S + h * k3S,
        I + h * k3I,
        R + h * k3R,
        beta,
        gamma,
        N
    )

    S_next = S + (h / 6.0) * (k1S + 2*k2S + 2*k3S + k4S)
    I_next = I + (h / 6.0) * (k1I + 2*k2I + 2*k3I + k4I)
    R_next = R + (h / 6.0) * (k1R + 2*k2R + 2*k3R + k4R)

    return S_next, I_next, R_next


def simulate_sir(S0, I0, R0, beta, gamma, N, days):
    """
    Simula o modelo SIR por uma quantidade de dias.
    """

    S_values = np.zeros(days)
    I_values = np.zeros(days)
    R_values = np.zeros(days)

    S_values[0] = S0
    I_values[0] = I0
    R_values[0] = R0

    S, I, R = S0, I0, R0

    for t in range(1, days):
        S, I, R = rk4_step(S, I, R, beta, gamma, N)
        S_values[t] = max(S, 0)
        I_values[t] = max(I, 0)
        R_values[t] = max(R, 0)

    return S_values, I_values, R_values


# ============================================================
# 3. FUNÇÃO DE ERRO PARA MÍNIMOS QUADRADOS
# ============================================================

def objective(params, S0, I0, R0, N, observed_I):
    """
    Função objetivo para ajuste por mínimos quadrados.

    Minimiza:

        mean((I_simulado - I_observado)^2)
    """

    beta, gamma = params

    if beta <= 0 or gamma <= 0:
        return np.inf

    days = len(observed_I)
    _, I_sim, _ = simulate_sir(S0, I0, R0, beta, gamma, N, days)

    return np.mean((I_sim - observed_I) ** 2)


# ============================================================
# 4. LEITURA E PREPARAÇÃO DOS DADOS
# ============================================================

def carregar_dados(path):
    """
    Lê o CSV tratado e organiza as colunas necessárias.
    """

    df = pd.read_csv(path)

    df.columns = [col.lower().strip() for col in df.columns]

    if "date" not in df.columns:
        raise ValueError("A coluna 'date' não foi encontrada no banco.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

   if "active_smooth" in df.columns:
    observed_I = df["active_smooth"].fillna(0).to_numpy()
elif "active" in df.columns:
    observed_I = df["active"].fillna(0).to_numpy()
elif "confirmed" in df.columns:
    observed_I = df["confirmed"].diff().fillna(0).clip(lower=0).to_numpy()
else:
    raise ValueError("Não foi encontrada coluna adequada para representar I(t).")

# corta o início da série até o primeiro valor positivo
inicio = np.argmax(observed_I > 0)

df = df.iloc[inicio:].reset_index(drop=True)
observed_I = observed_I[inicio:]

if "population" not in df.columns:
    raise ValueError("A coluna 'population' não foi encontrada no banco.")

N = float(df["population"].dropna().iloc[-1])

I0 = float(observed_I[0])

if "recovered_est" in df.columns:
    R0 = max(float(df["recovered_est"].fillna(0).iloc[0]), 0.0)
elif "recovered" in df.columns:
    R0 = max(float(df["recovered"].fillna(0).iloc[0]), 0.0)
else:
    R0 = 0.0

S0 = max(N - I0 - R0, 0.0)

return df, observed_I, S0, I0, R0, N


# ============================================================
# 5. GERAÇÃO DE GRÁFICOS
# ============================================================

def salvar_grafico_comparacao(datas, observed_I, I_sim):
    """
    Gera gráfico de infectados observados versus infectados simulados.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(datas, observed_I, label="Infectados observados")
    plt.plot(datas, I_sim, label="Infectados simulados - SIR/RK4")
    plt.xlabel("Data")
    plt.ylabel("Quantidade de infectados")
    plt.title("COVID-19 no RJ: dados reais x modelo SIR com RK4")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparacao_infectados.png"), dpi=300)
    plt.close()


def salvar_grafico_sir(datas, S_sim, I_sim, R_sim):
    """
    Gera gráfico das três curvas do modelo SIR.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(datas, S_sim, label="Suscetíveis S(t)")
    plt.plot(datas, I_sim, label="Infectados I(t)")
    plt.plot(datas, R_sim, label="Removidos/Recuperados R(t)")
    plt.xlabel("Data")
    plt.ylabel("População")
    plt.title("Evolução temporal do modelo SIR - RJ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "curvas_sir.png"), dpi=300)
    plt.close()


def salvar_grafico_erro(datas, observed_I, I_sim):
    """
    Gera gráfico do erro entre observado e simulado.
    """

    erro = observed_I - I_sim

    plt.figure(figsize=(12, 6))
    plt.plot(datas, erro, label="Erro: observado - simulado")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Data")
    plt.ylabel("Erro")
    plt.title("Erro do ajuste do modelo SIR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "erro_ajuste.png"), dpi=300)
    plt.close()


# ============================================================
# 6. EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    df, observed_I, S0, I0, R0, N = carregar_dados(DATA_PATH)

    print("Dados carregados com sucesso.")
    print(f"População N: {N:.0f}")
    print(f"S(0): {S0:.0f}")
    print(f"I(0): {I0:.0f}")
    print(f"R(0): {R0:.0f}")

    initial_guess = [0.3, 0.1]
    bounds = [(0.000001, 2.0), (0.000001, 2.0)]

    result = minimize(
        objective,
        initial_guess,
        args=(S0, I0, R0, N, observed_I),
        bounds=bounds,
        method="L-BFGS-B"
    )

    beta_opt, gamma_opt = result.x

    print("\nParâmetros ajustados:")
    print(f"beta  = {beta_opt:.6f}")
    print(f"gamma = {gamma_opt:.6f}")

    if gamma_opt > 0:
        R0_epidemiologico = beta_opt / gamma_opt
        print(f"R0 epidemiológico aproximado = {R0_epidemiologico:.4f}")

    days = len(observed_I)

    S_sim, I_sim, R_sim = simulate_sir(
        S0,
        I0,
        R0,
        beta_opt,
        gamma_opt,
        N,
        days
    )

    datas = df["date"]

    mse = np.mean((I_sim - observed_I) ** 2)
    rmse = np.sqrt(mse)

    print("\nMétricas de erro:")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")

    salvar_grafico_comparacao(datas, observed_I, I_sim)
    salvar_grafico_sir(datas, S_sim, I_sim, R_sim)
    salvar_grafico_erro(datas, observed_I, I_sim)

    resultado_df = pd.DataFrame({
        "date": datas,
        "observed_I": observed_I,
        "S_sim": S_sim,
        "I_sim": I_sim,
        "R_sim": R_sim,
        "erro_I": observed_I - I_sim
    })

    resultado_df.to_csv("resultado_sir_rk4_rj.csv", index=False)

    print("\nArquivos gerados:")
    print("- resultado_sir_rk4_rj.csv")
    print("- graficos/comparacao_infectados.png")
    print("- graficos/curvas_sir.png")
    print("- graficos/erro_ajuste.png")


if __name__ == "__main__":
    main()
