import pandas as pd
import os

def tratamento_dados(caminho_entrada, caminho_saida):
    print("Lendo o arquivo BRA.csv do COVID-19 Data Hub...")
    # low_memory=False evita os avisos de tipos misturados durante a leitura do arquivo
    df = pd.read_csv(caminho_entrada, low_memory=False)
    
    # 1. Converte a coluna de data para o tipo correto
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Filtra diretamente os dados NACIONAIS e do Brasil
    # administrative_area_level == 1 pega o país inteiro
    # iso_alpha_3 == 'BRA' garante que não pegaremos outros países caso a base seja global
    df_brasil = df[(df['administrative_area_level'] == 1) & (df['iso_alpha_3'] == 'BRA')].copy()
    
    # 3. Ordena cronologicamente
    df_brasil = df_brasil.sort_values('date')
    
    # 4. Calcula casos e óbitos DIÁRIOS a partir dos dados acumulados (confirmed e deaths)
    df_brasil['casos_diarios'] = df_brasil['confirmed'].diff().fillna(0)
    df_brasil['obitos_diarios'] = df_brasil['deaths'].diff().fillna(0)
    
    # Previne valores negativos gerados por correções retroativas das secretarias de saúde
    df_brasil.loc[df_brasil['casos_diarios'] < 0, 'casos_diarios'] = 0
    df_brasil.loc[df_brasil['obitos_diarios'] < 0, 'obitos_diarios'] = 0
    
    # 5. Aplica a Média Móvel de 7 dias
    df_brasil['casos_mm7d'] = df_brasil['casos_diarios'].rolling(window=7, min_periods=1).mean()
    df_brasil['obitos_mm7d'] = df_brasil['obitos_diarios'].rolling(window=7, min_periods=1).mean()
    
    # 6. Salva o arquivo final
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    
    # Vamos salvar apenas as colunas úteis para o modelo SEIRD-V, ignorando o resto para deixar o CSV bem leve
    colunas_finais = ['date', 'confirmed', 'deaths', 'casos_diarios', 'obitos_diarios', 'casos_mm7d', 'obitos_mm7d']
    df_brasil[colunas_finais].to_csv(caminho_saida, index=False)
    
    print(f"Sucesso! Dados processados e salvos em: {caminho_saida}")
    return df_brasil

if __name__ == "__main__":
    caminho_entrada = "data/raw/BRA.csv"
    caminho_saida = "data/processed/covid_brasil_limpo.csv"
    
    tratamento_dados(caminho_entrada, caminho_saida)