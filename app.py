import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from Modelos import Modelo_Logistico as ML, Modelo_Gompertz as MG
from Modelos import Modelo_Logistico_Generalizado as MLG, Modelo_Gompertz_Modificado as MGM

st.markdown("""
# Modelagem Matemática
            
## Aplicação de modelos baseados em EDO's em experimentos com :blue[camudongos nudes]

### Doença: :red[Glioma humano]

### Grupos de tratamento: 
- Controle
- Droga
- Radiação  
- Droga+Radiação""")

st.text("Os dados experimentais utilizados neste trabalho foram obtidos a partir de uma base pública. Diponível em:")
st.link_button("Tumor Growth", "https://www.causeweb.org/tshs/tumor-growth/")

st.markdown("""### Modelos matemáticos utilizados:""")

st.markdown(""" #### Modelo Logístico""")
st.latex(r''' \begin{equation*} \begin{cases} \displaystyle \frac{dN}{dt} = r N \left(1 - \frac{N}{K} \right)  \\ 
         N(0) = n_0\end{cases}\end{equation*}''')

st.markdown("""#### Modelo de Gompertz""")
st.latex(r''' \begin{equation*} \begin{cases} \displaystyle \frac{dN}{dt} = r N \ln\left(\frac{K}{N} \right)  \\ 
         N(0) = n_0\end{cases}\end{equation*}''')

st.markdown("""#### Modelo Logístico Generalizado""")
st.latex(r''' \begin{equation*} \begin{cases} \displaystyle \frac{dN}{dt} = \frac{r}{\theta} N \left[1 - \left( \frac{N}{K} \right)^{\theta} \right] \\  \\ 
         N(0) = n_0\end{cases}\end{equation*}''')

st.markdown("""#### Modelo de Gompertz Modificado""")
st.latex(r''' \begin{equation*} \begin{cases} \displaystyle \frac{dN}{dt} = r \left( \frac{e}{K}\right) \ln\left(\frac{K}{N} N \right)  \\ \\ 
         \displaystyle N(0) = K \exp \left[-\exp \left( \frac{\lambda r e}{K}\right)\right]\end{cases}\end{equation*}''')

#=================================================================================================
#
#=================================================================================================


st.markdown(""" ### Dados dos Camudongo""")

# Criando caixa seletora com os grupos experimentais
option1 = st.selectbox("Escolha um Grupo:",
                      ("Controle", "Droga", "Radiação", "Droga+Radiação"), key="Grupos")

# Função para dados informativos:
# Média do Tumor, Volume Máximo, Dia do Óbito
@st.cache_data
def dados(option:str):
    dados_camudongos = pd.read_csv("Datasets/dados_dos_camundongos.csv")
    lista = []
    for i in dados_camudongos.index:
       id, volume, obito, media = dados_camudongos.loc[i].values
       if id == option:
           lista.append(media)
           lista.append(volume)
           lista.append(obito)
    media, volume, obito = lista
    dados = pd.DataFrame({"Média do Tumor":[media], "Volume Máximo":[volume], "Dia do Óbito":[obito]})
    return dados.round(2)

# Figura de exibição dos dados dos grupos experimentais
def figura1(option:str):
    fig = f"Figuras/Experimento/{option}.png"
    return fig

# Criando variaveis para identificação de cada grupo
id_Controle =["ID-"+str(101 + i) for i in range(8)]
id_Droga  =["ID-"+str(201 + i) for i in range(10)]
id_Radiacao =["ID-"+str(301 + i) for i in range(10)]
id_Droga_Radiacao =["ID-"+str(401 + i) for i in range(9)]

# Grupo de Controle
if option1 == "Controle":
    option2 = st.selectbox("Escolha o ID", id_Controle, key="Grupo Controle")
    st.dataframe(dados(option=option2), hide_index=True)
    st.image(figura1(option2))

# Grupo de Droga
elif option1 == "Droga":
    option3 = st.selectbox("Escolha o ID", id_Droga, key="Grupo Droga")
    st.dataframe(dados(option=option3), hide_index=True)
    st.image(figura1(option3))

# Grupo de Radiação
elif option1 == "Radiação":
    option4 = st.selectbox("Escolha o ID", id_Radiacao, key="Grupo Radiação")
    st.dataframe(dados(option=option4), hide_index=True)
    st.image(figura1(option4))

# Grupo de Droga + Radiação
elif option1 == "Droga+Radiação":
    option5 = st.selectbox("Escolha o ID", id_Droga_Radiacao, key="Grupo Droga + Radiação")
    st.dataframe(dados(option=option5), hide_index=True)
    st.image(figura1(option5))


#=================================================================================================
#
#=================================================================================================


st.markdown(""" ### Dados da estimação dos parâmetros dos Modelos """)

# Conjunto de dataset dos parâmetros estimados
@st.cache_data
def data_set_load():
    Logistico = pd.read_csv("Datasets/Dados_Logistico.csv")
    Gompertz = pd.read_csv("Datasets/Dados_Gompertz.csv")
    Logistico_Generalizado = pd.read_csv("Datasets/Dados_Logistico_Generalizado.csv")
    Gompertz_Modificado = pd.read_csv("Datasets/Dados_Gompertz_Modificado.csv")
    return [Logistico, Gompertz, Logistico_Generalizado, Gompertz_Modificado]

# Carregandos os dados
df_Logistico, df_Gompertz, df_Log_Gen, df_Gomp_Mod = data_set_load()

# Função para carregar os dados do modelos escolhido
def modelos(obj:object):
    dic = obj
    return st.dataframe(pd.DataFrame(dic), hide_index=True)

# Criando caixa seletora dos modelos estudados
model = st.selectbox("Escolha o Modelo",
                     ("Logístico", "Gompertz", "Logístico Generalizado", "Gompertz Modificado"),
                     key="modelos")

# Parâmetros do Modelo Logístico
if model == "Logístico":
    grupo = st.selectbox("Escolha um Grupo", ("Controle", "Droga", "Radiação", "Droga+Radiação"),  key="Prametros Logístico")
    if grupo == "Controle": # 0 - 7
        modelos(ML(df_Logistico).lista_de_parametros(inicial=0, final=7))

    elif grupo == "Droga": # 8 - 17
        modelos(ML(df_Logistico).lista_de_parametros(8,17))

    elif grupo == "Radiação": # 18-27
        modelos(ML(df_Logistico).lista_de_parametros(18,27))
    
    elif grupo == "Droga+Radiação": # 28-36
        modelos(ML(df_Logistico).lista_de_parametros(28,38))

# Parâmetros do Modelo de Gompertz
elif model == "Gompertz":
    grupo = st.selectbox("Escolha um Grupo", ("Controle", "Droga", "Radiação", "Droga+Radiação"), key="Prametros Gompertz")
    if grupo == "Controle": # 0 - 7
        modelos(MG(df_Gompertz).lista_de_parametros(inicial=0, final=7))

    elif grupo == "Droga": # 8 - 17
        modelos(MG(df_Gompertz).lista_de_parametros(8,17))

    elif grupo == "Radiação": # 18-27
        modelos(MG(df_Gompertz).lista_de_parametros(18,27))
    
    elif grupo == "Droga+Radiação": # 28-36
        modelos(MG(df_Gompertz).lista_de_parametros(28,38))

# Parâmetros do Modelo Logístico Generalizado
if model == "Logístico Generalizado":
    grupo = st.selectbox("Escolha um Grupo", ("Controle", "Droga", "Radiação", "Droga+Radiação"),  key="Prametros Logístico Generalizado")
    if grupo == "Controle": # 0 - 7
        modelos(MLG(df_Log_Gen).lista_de_parametros(inicial=0, final=7))

    elif grupo == "Droga": # 8 - 17
        modelos(MLG(df_Log_Gen).lista_de_parametros(8,17))

    elif grupo == "Radiação": # 18-27
        modelos(MLG(df_Log_Gen).lista_de_parametros(18,27))
    
    elif grupo == "Droga+Radiação": # 28-36
        modelos(MLG(df_Log_Gen).lista_de_parametros(28,38))

# Parâmetros do Modelo Gompertz Modificado
if model == "Gompertz Modificado":
    grupo = st.selectbox("Escolha um Grupo", ("Controle", "Droga", "Radiação", "Droga+Radiação"),  key="Prametros Gompertz Modificado")
    if grupo == "Controle": # 0 - 7
        modelos(MGM(df_Gomp_Mod).lista_de_parametros(inicial=0, final=7))

    elif grupo == "Droga": # 8 - 17
        modelos(MGM(df_Gomp_Mod).lista_de_parametros(8,17))

    elif grupo == "Radiação": # 18-27
        modelos(MGM(df_Gomp_Mod).lista_de_parametros(18,27))
    
    elif grupo == "Droga+Radiação": # 28-36
        modelos(MGM(df_Gomp_Mod).lista_de_parametros(28,38)) 

#=================================================================================================
#
#=================================================================================================


st.markdown("""### Curvas Ajustadas""")

# Criando caixa seletora para visualizar as curvas do modelo escolhido
model = st.selectbox("Escolha o Modelo",
                     ("Logístico", "Gompertz", "Logístico Generalizado", "Gompertz Modificado"),
                     key="modelos simulação")

# Curvas do Modelo Logístico
def figura2(pasta:str, option:str): # Função para pegar as imagens da pasta
    fig = f"Figuras/{pasta}/{option}.png"
    return fig

# Função Criadora do botão
def grupo(chave:str):
    return st.selectbox("Escolha um Grupo", ("Controle", "Droga", "Radiação", "Droga+Radiação"),  key=chave)

# Função Criadora da imagem
def construcao(id:list, pasta:str):
    escolha = st.selectbox("Selecione um ID", id, key="Curvas Controle")
    return st.image(figura2(pasta, option=escolha))

if model == "Logístico":
    grupo = st.selectbox("Escolha um Grupo", ("Controle", "Droga", "Radiação", "Droga+Radiação"),  key="Simulçao Logístico")
    if grupo == "Controle": # 0 - 7
        escolha = st.selectbox("Selecione um ID", id_Controle, key="Curvas Controle")
        st.image(figura2(pasta="Logistico", option=escolha))

    elif grupo == "Droga": # 8 - 17
        escolha = st.selectbox("Selecione um ID", id_Droga, key="Curvas Droga")
        st.image(figura2(pasta="Logistico", option=escolha))

    elif grupo == "Radiação": # 18-27
       escolha = st.selectbox("Selecione um ID", id_Radiacao, key="Curvas Radiação")
       st.image(figura2(pasta="Logistico", option=escolha))
    
    elif grupo == "Droga+Radiação": # 28-36
        escolha = st.selectbox("Selecione um ID", id_Droga_Radiacao, key="Curvas Droga+Radiação")
        st.image(figura2(pasta="Logistico", option=escolha))

elif model == "Gompertz":
    grupo = grupo(chave="Curvas Gompertz")
    if grupo == "Controle":
        construcao(id=id_Controle, pasta="Gompertz")
    elif grupo == "Droga":
        construcao(id=id_Droga, pasta="Gompertz")
    elif grupo == "Radiação":
        construcao(id=id_Radiacao, pasta="Gompertz")
    elif grupo == "Droga+Radiação":
        construcao(id=id_Droga_Radiacao, pasta="Gompertz")

elif model == "Logístico Generalizado":
    grupo = grupo(chave="Curvas Logístico Generalizado")
    if grupo == "Controle":
        construcao(id=id_Controle, pasta="Logistico_Generalizado")
    elif grupo == "Droga":
        construcao(id=id_Droga, pasta="Logistico_Generalizado")
    elif grupo == "Radiação":
        construcao(id=id_Radiacao, pasta="Logistico_Generalizado")
    elif grupo == "Droga+Radiação":
        construcao(id=id_Droga_Radiacao, pasta="Logistico_Generalizado")

elif model == "Gompertz Modificado":
    grupo = grupo(chave="Curvas Gompertz Modificado")
    if grupo == "Controle":
        construcao(id=id_Controle, pasta="Gompertz_Modificado")
    elif grupo == "Droga":
        construcao(id=id_Droga, pasta="Gompertz_Modificado")
    elif grupo == "Radiação":
        construcao(id=id_Radiacao, pasta="Gompertz_Modificado")
    elif grupo == "Droga+Radiação":
        construcao(id=id_Droga_Radiacao, pasta="Gompertz_Modificado")

#=================================================================================================
#
#=================================================================================================

# RMSEs
st.markdown("""#### Comparação dos RMSEs de cada modelo""")
controle, droga, radiacao, drogra_radiacao = st.columns(4)

if controle.button("Grupo de Controle", width="stretch"):
    st.image(figura2(pasta="RMSE", option="RMSE_Controle"))

if droga.button("Grupo de Droga", width="stretch"):
    st.image(figura2(pasta="RMSE", option="RMSE_Droga"))

if radiacao.button("Grupo de Radiação", width="stretch"):
    st.image(figura2(pasta="RMSE", option="RMSE_Radiação"))

if drogra_radiacao.button("Grupo de Droga+Radiação", width="stretch"):
    st.image(figura2(pasta="RMSE", option="RMSE_Droga_Radiação"))
















