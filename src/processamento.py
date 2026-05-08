import pandas as pd
import os

def tratamento_dados(caminho_entrada, caminho_saida):
    df = pd.read_csv(caminho_entrada)
    
    df['date'] = pd.to_datetime(df['date'])
    
    df_estados = df[df['place_type'] == 'state'].copy()
    
    df_brasil = df_estados.groupby('date')[['confirmed', 'deaths']].sum().reset_index()
    df_brasil = df_brasil.sort_values('date')
    
    df_brasil['casos_diarios'] = df_brasil['confirmed'].diff().fillna(0)
    df_brasil['obitos_diarios'] = df_brasil['deaths'].diff().fillna(0)
    
    df_brasil.loc[df_brasil['casos_diarios'] < 0, 'casos_diarios'] = 0
    df_brasil.loc[df_brasil['obitos_diarios'] < 0, 'obitos_diarios'] = 0
    
    df_brasil['casos_mm7d'] = df_brasil['casos_diarios'].rolling(window = 7, min_periods = 1).mean()
    df_brasil['obitos_mm7d'] = df_brasil['obitos_diarios'].rolling(window = 7, min_periods = 1).mean()
    
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    df_brasil.to_csv(caminho_saida, index=False)
    
    return df_brasil