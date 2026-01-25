import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from Modelos import Modelo_Logistico as ML, Modelo_Gompertz as MG
from Modelos import Modelo_Logistico_Generalizado as MLG, Modelo_Gompertz_Modificado as MGM
from Modelos import Datasets as data

st.markdown("""
# Modelagem Matemática
            
## Aplicação de modelos baseados em EDO's em experimentos com :blue[camudongos nudes]

### Doença: :red[Glioma humano]

### Grupos de tratamento: 
- Controle
- Droga
- Radiação  
- Droga+Radiação

### Modelos matemáticos utilizados: """)

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

option1 = st.selectbox("Escolha um Grupo:",
                      ("Controle", "Droga", "Radiação", "Droga+Radiação"), key="Grupos")

# Função para dados informativos
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

# Figura de exibição dos dados
def figura1(option:str):
    fig = f"Figuras/Experimento/{option}.png"
    return fig

# Grupo de Controle
if option1 == "Controle":
    id =["ID-"+str(101 + i) for i in range(8)]
    option2 = st.selectbox("Escolha o ID", id, key="Grupo Controle")
    st.dataframe(dados(option=option2), hide_index=True)
    st.image(figura1(option2))

# Grupo de Droga
elif option1 == "Droga":
    id = ("ID-"+str(201 + i) for i in range(10))
    option3 = st.selectbox("Escolha o ID", id, key="Grupo Droga")
    st.dataframe(dados(option=option3), hide_index=True)
    st.image(figura1(option3))

# Grupo de Radiação
elif option1 == "Radiação":
    id = ("ID-"+str(301 + i) for i in range(10))
    option4 = st.selectbox("Escolha o ID", id, key="Grupo Radiação")
    st.dataframe(dados(option=option4), hide_index=True)
    st.image(figura1(option4))

# Grupo de Droga + Radiação
elif option1 == "Droga+Radiação":
    id = ("ID-"+str(401 + i) for i in range(9))
    option5 = st.selectbox("Escolha o ID", id, key="Grupo Droga + Radiação")
    st.dataframe(dados(option=option5), hide_index=True)
    st.image(figura1(option5))


#=================================================================================================
#
#=================================================================================================


st.markdown(""" ### Dados da estimação dos parâmetros dos Modelos """)

# Conjunto de dataset dos parâmetros estimados
df_Logistico = data().data_set_load("Modelo_Logistico")
df_Gompertz = data().data_set_load("Modelo_Gompertz")
df_Log_Gen = data().data_set_load("Modelo_Logistico_Generalizado")
df_Gomp_Mod = data().data_set_load("Modelo_Gompertz_Modificado")


# Função Objeto
def modelos(obj:object):
    dic = obj
    return st.dataframe(pd.DataFrame(dic), hide_index=True)

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
        id = ["ID-"+str(101 + i) for i in range(8)]
        escolha = st.selectbox("Selecione um ID", id, key="Curvas Controle")
        st.image(figura2(pasta="Logistico", option=escolha))

    elif grupo == "Droga": # 8 - 17
        id = ["ID-"+str(201 + i) for i in range(10)]
        escolha = st.selectbox("Selecione um ID", id, key="Curvas Droga")
        st.image(figura2(pasta="Logistico", option=escolha))

    elif grupo == "Radiação": # 18-27
       id = ["ID-"+str(301 + i) for i in range(10)]
       escolha = st.selectbox("Selecione um ID", id, key="Curvas Radiação")
       st.image(figura2(pasta="Logistico", option=escolha))
    
    elif grupo == "Droga+Radiação": # 28-36
        id = ["ID-"+str(401 + i) for i in range(9)]
        escolha = st.selectbox("Selecione um ID", id, key="Curvas Droga+Radiação")
        st.image(figura2(pasta="Logistico", option=escolha))

elif model == "Gompertz":
    grupo = grupo(chave="Curvas Gompertz")
    if grupo == "Controle":
        id = ["ID-"+str(101 + i) for i in range(8)]
        construcao(id=id, pasta="Gompertz")
    elif grupo == "Droga":
        id = ["ID-"+str(201 + i) for i in range(10)]
        construcao(id=id, pasta="Gompertz")
    elif grupo == "Radiação":
        id = ["ID-"+str(301 + i) for i in range(10)]
        construcao(id=id, pasta="Gompertz")
    elif grupo == "Droga+Radiação":
        id = ["ID-"+str(401 + i) for i in range(9)]
        construcao(id=id, pasta="Gompertz")

elif model == "Logístico Generalizado":
    grupo = grupo(chave="Curvas Logístico Generalizado")
    if grupo == "Controle":
        id = ["ID-"+str(101 + i) for i in range(8)]
        construcao(id=id, pasta="Logistico_Generalizado")
    elif grupo == "Droga":
        id = ["ID-"+str(201 + i) for i in range(10)]
        construcao(id=id, pasta="Logistico_Generalizado")
    elif grupo == "Radiação":
        id = ["ID-"+str(301 + i) for i in range(10)]
        construcao(id=id, pasta="Logistico_Generalizado")
    elif grupo == "Droga+Radiação":
        id = ["ID-"+str(401 + i) for i in range(9)]
        construcao(id=id, pasta="Logistico_Generalizado")

elif model == "Gompertz Modificado":
    grupo = grupo(chave="Curvas Gompertz Modificado")
    if grupo == "Controle":
        id = ["ID-"+str(101 + i) for i in range(8)]
        construcao(id=id, pasta="Gompertz_Modificado")
    elif grupo == "Droga":
        id = ["ID-"+str(201 + i) for i in range(10)]
        construcao(id=id, pasta="Gompertz_Modificado")
    elif grupo == "Radiação":
        id = ["ID-"+str(301 + i) for i in range(10)]
        construcao(id=id, pasta="Gompertz_Modificado")
    elif grupo == "Droga+Radiação":
        id = ["ID-"+str(401 + i) for i in range(9)]
        construcao(id=id, pasta="Gompertz_Modificado")


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
















