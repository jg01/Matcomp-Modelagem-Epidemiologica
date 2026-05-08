import pandas as pd
import matplotlib.pyplot as plt
import os

def plotar_comparacao(caminho_dados):
    print(f"A carregar dados de: {caminho_dados}")
    df = pd.read_csv(caminho_dados)
    
    # Garantir que a coluna date é interpretada como data
    df['date'] = pd.to_datetime(df['date'])
    
    # Criar a figura com um tamanho adequado para apresentações (widescreen)
    plt.figure(figsize=(14, 7))
    
    # 1. Plotar os dados RAW (brutos) como barras cinzentas semi-transparentes
    plt.bar(df['date'], df['casos_diarios'], color='gray', alpha=0.4, label='Casos Diários (Raw / Ruído)')
    
    # 2. Plotar os dados PROCESSED (média móvel) como uma linha vermelha contínua
    plt.plot(df['date'], df['casos_mm7d'], color='red', linewidth=2.5, label='Média Móvel 7 Dias (Processed / Sinal)')
    
    # Formatação estética e académica do gráfico
    plt.title('Casos Diários de COVID-19 no Brasil: Impacto do Tratamento de Dados', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Número de Casos Registados', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Guardar o gráfico como imagem antes de o mostrar
    caminho_imagem = 'data/processed/grafico_casos_brasil.png'
    plt.savefig(caminho_imagem, dpi=300)
    print(f"Gráfico guardado com sucesso em: {caminho_imagem}")
    
    # Apresentar a janela interativa com o gráfico
    plt.show()

if __name__ == "__main__":
    caminho_entrada = "data/processed/covid_brasil_limpo.csv"
    
    # Verifica se o ficheiro processado já existe antes de tentar plotar
    if os.path.exists(caminho_entrada):
        plotar_comparacao(caminho_entrada)
    else:
        print(f"Erro: O ficheiro {caminho_entrada} não foi encontrado. Execute o processamento.py primeiro.")