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
from datetime import date, timedelta
import matplotlib.pyplot as plt

hoje = date.today()

lista_de_acoes = ["ABEV3.SA", "ARZZ3.SA", "B3SA3.SA", "BBDC3.SA", "BBSE3.SA", "EGIE3.SA", "ENBR3.SA", "EZTC3.SA", "GRND3.SA", "ITSA3.SA", "ITUB3.SA", "LEVE3.SA", "LREN3.SA", "MDIA3.SA", "ODPV3.SA", "PSSA3.SA", "RADL3.SA", "SLCE3.SA", "TOTS3.SA", "VIVT3.SA", "VULC3.SA", "WEGE3.SA"]

for acao in lista_de_acoes:
    df = web.DataReader(acao, "yahoo", hoje - pd.DateOffset(months=3), hoje)
    df = df.iloc[::-1] # inverter o dataframe para que o primeiro dia seja o mais recente
    # df.to_excel("dados.xlsx", sheet_name=acao)

    lista_medias = []
    for i in range(0, 90, 5):   # calcular a média até 90 dias para trás
        media = df.loc[hoje-timedelta(i):hoje-timedelta(i+5), "Adj Close"].mean()
        lista_medias.append(media)

    media = 0
    lista_variacao = []
    for media_anterior in lista_medias:
        if media == 0:
            media = media_anterior
        else:
            variacao = (media - media_anterior) / media_anterior * 100
            # formatar a variacao para float com 5 casas decimais
            variacao = float("{:.5f}".format(variacao))
            lista_variacao.append(variacao)
            media = media_anterior
    
    print(f"lista de medias: {lista_medias}")
    print(f"lista de variacoes: {lista_variacao}")

    plt.plot(lista_variacao)
    plt.plot(lista_medias)
    plt.show()