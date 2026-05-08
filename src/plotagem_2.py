import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def plotar_comparacao_zoom(caminho_dados):
    print(f"A carregar dados de: {caminho_dados}")
    df = pd.read_csv(caminho_dados)
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. O "Zoom": Vamos filtrar apenas a primeira grande onda (Março a Agosto de 2020)
    # Isso permite que os pontos fiquem visíveis e não se sobreponham.
    df_zoom = df[(df['date'] >= '2020-03-01') & (df['date'] <= '2020-08-31')]
    
    plt.figure(figsize=(14, 7))
    
    # 2. Barras diárias
    plt.bar(df_zoom['date'], df_zoom['casos_diarios'], color='gray', alpha=0.4, label='Casos Diários (Raw)')
    
    # 3. Linha com PONTOS: Adicionamos marker='o' (bolinhas) e markersize=4
    plt.plot(df_zoom['date'], df_zoom['casos_mm7d'], color='red', linewidth=2, 
             marker='o', markersize=4, label='Média Móvel 7 Dias')
    
    plt.title('Zoom na 1ª Onda: Casos Diários de COVID-19 vs Média Móvel', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Número de Casos', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    
    # 4. Forçando a formatação detalhada do Eixo X
    ax = plt.gca()
    # Coloca um rótulo principal a cada mês
    ax.xaxis.set_major_locator(mdates.MonthLocator()) 
    # Coloca pequenas marcações a cada 7 dias para provar a granularidade
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7)) 
    # Formata o texto para "Mês/Ano" (ex: Mar/2020)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
    
    plt.xticks(rotation=45) # Gira o texto para não encavalar
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Salva a imagem nova
    caminho_imagem = 'data/processed/grafico_zoom_pontos.png'
    plt.savefig(caminho_imagem, dpi=300)
    print(f"Gráfico detalhado guardado em: {caminho_imagem}")
    
    plt.show()

if __name__ == "__main__":
    caminho_entrada = "data/processed/covid_brasil_limpo.csv"
    if os.path.exists(caminho_entrada):
        plotar_comparacao_zoom(caminho_entrada)
    else:
        print("Erro: Ficheiro não encontrado.")