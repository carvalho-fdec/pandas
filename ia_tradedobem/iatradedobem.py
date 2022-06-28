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
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

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
    print(f"{acao} em {data.strftime('%d/%m/%Y')}...", end="")
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

# função para avaliar um modelo de classificação
def avaliar(y_teste, previsoes, nome_modelo):
    print(nome_modelo)
    report = classification_report(y_teste, previsoes)
    print(report)
    cf_matrix = pd.DataFrame(confusion_matrix(y_teste, previsoes), index=["Nao Comprar", "Comprar"], columns=["Nao Comprar", "Comprar"])
    sns.heatmap(cf_matrix, annot=True, cmap="Blues", fmt=',')
    # plt.show()
    print("#" * 50)

# função para gravar arquivo excel com a variação percencual nos 3 meses anteriores de todas as ações na data dos trades feitos
def gerar_arquivo_dados(arq_trades):   
    df = pd.DataFrame()
    df_todos = pd.DataFrame()
    trades_feitos = pd.read_excel(arq_trades)
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

    #tratar df_todos
    df_todos = df_todos.dropna("columns")
    df_todos = df_todos.reset_index(drop=True)
    # print(df_todos)
    df_todos.to_excel("dados.xlsx",sheet_name="dados", index=False)

def correlacoes(df_todos):
    correlacoes = df_todos.corr()

    ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(correlacoes, cmap="Wistia", ax=ax)
    # plt.show()
    correlacoes_encontradas = []
    for coluna in correlacoes:
        for linha in correlacoes.index:
            if linha != coluna:
                valor = abs(correlacoes.loc[linha, coluna])
                if valor > 0.8 and (coluna, linha, valor) not in correlacoes_encontradas:
                    correlacoes_encontradas.append((linha, coluna, valor))
                    print(f"Correlação Encontrada: {linha} e {coluna}. Valor: {valor}")
    print(f'\n\nTotal de Correlações Encontradas: {len(correlacoes_encontradas)}')

def ajustar_scaler(tabela_original):
    scaler = StandardScaler()
    tabela_auxiliar = tabela_original.drop("Decisao", axis=1)
    
    tabela_auxiliar = pd.DataFrame(scaler.fit_transform(tabela_auxiliar), tabela_auxiliar.index, tabela_auxiliar.columns)
    tabela_auxiliar["Decisao"] = tabela_original["Decisao"]
    return tabela_auxiliar

def ia_tradedobem (lista_de_acoes):
    # criar uma lista de ações com minhas últimas ações compradas para trade com ação e data da compra
    # criar um histórico dos últimos 3 meses desta ação em percentual
    # criar uma nova coluna no data frame da análise da ação com uma coluna decisão indicando compra (= 1) ou não compra (= 0)
    # criar um histórico dos últimos 3 meses das demais ações e na coluna decisão indicando não compra (0)
    # rodar o modelo dummy no data frame da análise da ação
    # rodar os demais modelos no data frame da análise da ação

    print("\n\nINICIO IA TRADE...") # , end="")

    # ESTÁ DANDO ERRO NA EXECUÇÃO APÓS COMENTAR AS LINHAS ACIMA
    # EXECUTAR COM TODAS AS AÇÕES PRA GRAVAR O ARQUIVO "dados.xlsx" COM TODOS OS TRADES
    # APÓS SO LER O ARQUIVO TRADE E NAO LER NO YAHOO FINANCE

    # gerar_arquivo_dados("trades.xlsx")
    df_todos = pd.DataFrame()
    df_todos = pd.read_excel("dados.xlsx",sheet_name="dados")
    df_todos = df_todos * 100
    df_todos['Decisao'] = (df_todos['Decisao']/100).astype(int)

    # correlacoes(df_todos)

    # treinar uma arvore de decisao e pegar as caracteristicas mais importantes dela

    modelo = ExtraTreesClassifier(random_state=1)
    x = df_todos.drop("Decisao", axis=1)
    y = df_todos["Decisao"]
    modelo.fit(x, y)

    caracteristicas_importantes = pd.DataFrame(modelo.feature_importances_, x.columns).sort_values(by=0, ascending=False)
    # print(caracteristicas_importantes)
    top10 = list(caracteristicas_importantes.index)[:10]
    # print(top10)
   


    nova_base_dados = ajustar_scaler(df_todos)
    top10.append("Decisao") 

    nova_base_dados = nova_base_dados[top10].reset_index(drop=True)
    # print(nova_base_dados)
   
    df_todos = nova_base_dados

    # separação dos dados em treino e teste
    x = df_todos.drop("Decisao", axis=1)
    y = df_todos["Decisao"]
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, random_state=1)

    # criar um modelo dummy
    modelo_dummy = DummyClassifier(strategy="stratified", random_state=2)
    modelo_dummy.fit(x_treino, y_treino)
    previsao_dummy = modelo_dummy.predict(x_teste)
    avaliar(y_teste, previsao_dummy, "Dummy")

    #demais mdelos
    modelos = {
        "AdaBoost": AdaBoostClassifier(random_state=1),
        "DecisionTree": DecisionTreeClassifier(random_state=1),
        "RandomForest": RandomForestClassifier(random_state=1),
        "ExtraTree": ExtraTreesClassifier(random_state=1),
        "GradientBoost": GradientBoostingClassifier(random_state=1),
        "KNN": KNeighborsClassifier(),
        "LogisticRegression": LogisticRegression(random_state=1),
        "NaiveBayes": GaussianNB(),
        "SVM": SVC(random_state=1),
        "RedeNeural": MLPClassifier(random_state=1, max_iter=400),
    }    
    
    for nome_modelo in modelos:
        modelo = modelos[nome_modelo]
        modelo.fit(x_treino, y_treino)
        previsoes = modelo.predict(x_teste)
        avaliar(y_teste, previsoes, nome_modelo)
        modelos[nome_modelo] = modelo

    # treinamento do modelo com o tuning
    modelo_final = modelos["RandomForest"] # melhor modelo para o tuning

    n_estimators = range(10, 251, 30)
    max_features = list(range(2, 11, 2))
    max_features.append('auto')
    min_samples_split = range(2, 11, 2)

    precision2_score = make_scorer(precision_score, labels=[1], average='macro')

    grid = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid={
                'n_estimators': n_estimators,
                'max_features': max_features,
                'min_samples_split': min_samples_split,
                'random_state': [1],
            },
            scoring=precision2_score,
    )
    

    resultado_grid = grid.fit(x_treino, y_treino)
    modelo_tunado = resultado_grid.best_estimator_
    previsoes = modelo_tunado.predict(x_teste)
    avaliar(y_teste, previsoes, "RandomForest Tunado")

    # aplicar o modelo tunado na lista de ações com data de hoje para prever a melhor ação para compra
    df_atual = pd.DataFrame()
    hoje = date.today()
    for acao in lista_de_acoes:
        df = fazer_df(acao, hoje, 0) 
        df_atual = df_atual.append(df)
    df_atual = df_atual.dropna("columns")
    # df_atual = df_atual.drop("Decisao", axis=1)
    # print(f"\n\ncolunas 1 drop: {df_atual.columns} e top10: {top10}")
    df_atual = df_atual[top10].reset_index(drop=True)
    # print(f"\n\ncolunas 2 drop: {df_atual.columns} e top10: {top10}")
    ajustar_scaler(df_atual)
    # print(f"\n\ncolunas scaler: {df_atual.columns} e top10: {top10}")
    df_atual = df_atual.drop("Decisao", axis=1)
    # print(df_atual)
    # return

    previsao_atual = modelo_tunado.predict(df_atual)
    print("\n\nPREVISÃO ATUAL ({hoje}):")
    for i, acao in enumerate(lista_de_acoes):
        if previsao_atual[i] == 1:
            print(acao)


if __name__ == "__main__":
    
    lista_de_acoes = ["ABEV3.SA", "ARZZ3.SA", "B3SA3.SA", "BBDC3.SA", "BBSE3.SA", "EGIE3.SA", "ENBR3.SA", "EZTC3.SA", "GRND3.SA", "ITSA3.SA", "ITUB3.SA", "LEVE3.SA", "LREN3.SA", "MDIA3.SA", "ODPV3.SA", "PSSA3.SA", "RADL3.SA", "SLCE3.SA", "TOTS3.SA", "VIVT3.SA", "VULC3.SA", "WEGE3.SA"]

    # lista_de_acoes = ["ABEV3.SA", "ARZZ3.SA", "B3SA3.SA", "BBDC3.SA"]
    
    # meu_tradedobem(lista_de_acoes)
    ia_tradedobem(lista_de_acoes)
