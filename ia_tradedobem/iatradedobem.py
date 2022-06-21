"""
Projeto usando Inteligência Artificial para escolher a melhor ação para comprar de uma lista pré-definida.
IA project to choose the best stock to buy on a list of selected stocks.
"""

# criar a lista de ações
# ler o histórico de preços de cada ação nos últimos 3 meses
# criar uma análise com base nas médias de preços de cada ação por semana, verificando se a média está subindo ou descendo, e qual percentual
# criar uma análise com base em inteligência artificial
# comprar as duas análises e escolher a melhor

from cmath import nan
import pandas as pd
import pandas_datareader.data as web
from datetime import date, timedelta
import matplotlib.pyplot as plt

def meu_tradedobem (lista_de_acoes):
    
    hoje = date.today()
    dict_acoes = {}

    print("INICIO MEU TRADE...", end="")

    for acao in lista_de_acoes:
        df = web.DataReader(acao, "yahoo", hoje - pd.DateOffset(months=3), hoje)
        df = df.iloc[::-1] # inverter o dataframe para que o primeiro dia seja o mais recente
        # df.to_excel("dados.xlsx",sheet_name=acao)

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

        print(f"{acao} ok.", end="")
    
    # dict_acoes = sorted(dict_acoes.items(), key=lambda x: x[1])
    dict_acoes = sorted(dict_acoes, key=dict_acoes.get)
    print("\n\nMEU TRADE DO BEM: ", end="")
    print(*dict_acoes, sep=' > ')


# função para criar um data frame com o histórico percentual (3 meses) de preços de uma ação
def fazer_df (acao, data, decisao):
    print(f"{acao}...", end="")
    data_compra = data
    try:
        df = web.DataReader(acao, "yahoo", data_compra - pd.DateOffset(months=3), data_compra)
    except:
        df = pd.DataFrame()
    else:
        df["Variacao"] = df["Adj Close"]/df["Adj Close"].shift(1) - 1
        df = df[::-1] # inverter o dataframe para que o primeiro dia seja o mais recente
        df = df["Variacao"].dropna().reset_index(drop=True).T
        df["Decisao"] = decisao
    return df


def ia_tradedobem (lista_de_acoes):
    # criar uma lista de ações com minhas últimas ações compradas para trade com ação e data da compra
    # criar um histórico dos últimos 3 meses desta ação em percentual
    # criar uma nova coluna no data frame da análise da ação com uma coluna decisão indicando compra (= 1) ou não compra (= 0)
    # criar um histórico dos últimos 3 meses das demais ações e na coluna decisão indicando não compra (0)
    # rodar o modelo dummy no data frame da análise da ação
    # rodar os demais modelos no data frame da análise da ação

    print("\n\nINICIO IA TRADE...") # , end="")

    df = pd.DataFrame()
    df_todos = pd.DataFrame()
    trades_feitos = pd.read_excel("tradesmenor.xlsx")
    for trade in trades_feitos.itertuples():
        acao = trade.AÇÃO
        acao = acao + ".SA"
        data_compra = trade.DATA
        data_compra = data_compra.strftime('%m/%d/%Y')
        data_compra = pd.to_datetime(data_compra)
        df = fazer_df(acao, data_compra, 1) # decisao 1 = compra
        df_todos = df_todos.append(df)
        for acoes in lista_de_acoes:
            if acoes != acao:
                df = fazer_df(acoes, data_compra, 0) # decisao 0 = não compra
                df_todos = df_todos.append(df)
        print(df_todos)

    #tratar df_todos
    df_todos = df_todos.dropna("columns")
    df_todos.to_excel("dados.xlsx",sheet_name="dados", index=False)



if __name__ == "__main__":
    
    # lista_de_acoes = ["ABEV3.SA", "ARZZ3.SA", "B3SA3.SA", "BBDC3.SA", "BBSE3.SA", "EGIE3.SA", "ENBR3.SA", "EZTC3.SA", "GRND3.SA", "ITSA3.SA", "ITUB3.SA", "LEVE3.SA", "LREN3.SA", "MDIA3.SA", "ODPV3.SA", "PSSA3.SA", "RADL3.SA", "SLCE3.SA", "TOTS3.SA", "VIVT3.SA", "VULC3.SA", "WEGE3.SA"]

    lista_de_acoes = ["ABEV3.SA", "ARZZ3.SA", "B3SA3.SA", "BBDC3.SA"]
    
    # meu_tradedobem(lista_de_acoes)
    ia_tradedobem(lista_de_acoes)
