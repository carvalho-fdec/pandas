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

lista_de_acoes = ["ABEV3.SA", "ARZZ3.SA", "B3SA3.SA", "BBDC3.SA", "BBSE3.SA", "EGIE3.SA", "ENBR3.SA", "EZTC3.SA", "GRND3.SA", "ITSA3.SA", "ITUB3.SA", "LEVE3.SA", "LREN3.SA", "MDIA3.SA", "ODPV3.SA", "PSSA3.SA", "RADL3.SA", "SLCE3.SA", "TOTS3.SA", "VIVT3.SA", "VULC3.SA", "WEGE3.SA"]


def meu_tradedobem (lista_de_acoes):
    
    hoje = date.today()
    dict_acoes = {}

    print("INICIO...", end="")

    for acao in lista_de_acoes:
        df = web.DataReader(acao, "yahoo", hoje - pd.DateOffset(months=3), hoje)
        df = df.iloc[::-1] # inverter o dataframe para que o primeiro dia seja o mais recente
        # df.to_excel("dados.xlsx", sheet_name=acao)

        lista_medias = []
        lista_percentual = []
        valor_acao_agora = df.loc[hoje.strftime("%m/%d/%Y"), "Adj Close"].item()

        for i in range(0, 90, 5):   # calcular a média até 90 dias para trás de 5 em 5 dias
            media = df.loc[hoje-timedelta(i):hoje-timedelta(i+5), "Adj Close"].mean()
            lista_medias.append(media)
            percentual = (valor_acao_agora - media) / valor_acao_agora * 100
            percentual = float("{:.5f}".format(percentual))
            lista_percentual.append(percentual)
            print(".", end="")

        dict_acoes[acao] = sum(lista_percentual)

        print(f"{acao} ok...", end="")
    
    dict_acoes = sorted(dict_acoes.items(), key=lambda x: x[1])
    # dict_acoes = sorted(dict_acoes, key=dict_acoes.get)
    print(f"\n\nDict acoes ordenado: {dict_acoes}")


if __name__ == "__main__":
    meu_tradedobem(lista_de_acoes)
