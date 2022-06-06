"""
Projeto usando Inteligência Artificial para escolher a melhor ação para comprar de uma lista pré-definida.
IA project to choose the best stock to buy on a list of selected stocks.
"""

# criar a lista de ações
# ler o histórico de preços de cada ação nos últimos 3 meses
# criar uma análise com base nas médias de preços de cada ação por semana, verificando se a média está subindo ou descendo, e qual percentual
# criar uma análise com base em inteligência artificial
# comprar as duas análises e escolher a melhor

import pandas as pd
import pandas_datareader.data as web
from datetime import date

hoje = date.today()

lista_de_acoes = ["ABEV3.SA", "ARZZ3.SA", "B3SA3.SA", "BBDC3.SA", "BBSE3.SA", "EGIE3.SA", "ENBR3.SA", "EZTC3.SA", "GRND3.SA", "ITSA3.SA", "ITUB3.SA", "LEVE3.SA", "LREN3.SA", "MDIA3.SA", "ODPV3.SA", "PSSA3.SA", "RADL3.SA", "SLCE3.SA", "TOTS3.SA", "VIVT3.SA", "VULC3.SA", "WEGE3.SA"]

df = web.DataReader("ABEV3.SA", "yahoo", hoje - pd.DateOffset(months=3), hoje)
df = df.iloc[::-1] # inverter o dataframe para que o primeiro dia seja o mais recente

# criar um arquivo excell com os dados do df
df.to_excel("dados2.xlsx")

# calcular a média da coluna 'Adj Close' por semana
cont_aux = 0
soma = 0
cont_semana = 1
for i in df["Adj Close"]:
    print(i)
    if cont_aux == 5:
        media_semana = soma / 5
        print("Semana " + str(cont_semana) + ": " + str(media_semana))
        cont_semana += 1
        soma = 0
        cont_aux = 0
    soma += i
    cont_aux += 1

