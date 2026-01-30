import pandas as pd
import numpy as np
        
# Classe inicial para os parâmetros
class Parametros():
  def __init__(self, dataset:pd.DataFrame):
    self.dataset = dataset
    self.IDs = []
    self.Cs = []
    self.rs = []
    self.n0s = []
    self.RMSEs = []

  def lista_de_parametros(self, inicial:int, final:int):
    IDs = self.IDs
    Cs = self.Cs
    rs = self.rs
    n0s = self.n0s
    RMSEs = self.RMSEs
    
    for valor in range(len(self.dataset.loc[inicial:final])):
      ID, C, r, n0, RMSE = self.dataset.loc[inicial:final].values[valor]
      IDs.append(ID)
      Cs.append(C)
      rs.append(r)
      n0s.append(n0)
      RMSEs.append(RMSE)
    
    dic = {"ID":IDs, "C":Cs, "r":rs, "n0":n0s, "RMSE":RMSEs}

    return dic


class Modelo_Logistico(Parametros):
  
    def logistica(self, ID=str):
        K = 2000
        t = np.linspace(0,28, 100)
        lista = []
        for i in self.dataset.index:
            id, _, r, n0, _ = self.dataset.loc[i].values
            if id == ID:
                lista.append(r)
                lista.append(n0)
        r, n0 = lista
    
        return [t, K / (1 + ((K - n0) / n0) * np.exp(-r * t))]

class Modelo_Gompertz(Parametros):
        
    def gompertz(self, ID:str):
        K = 2000
        t = np.linspace(0,28, 100)
        lista = []
        for i in self.dataset.index:
            id, _, r, n0, _ = self.dataset.loc[i].values
            if id == ID:
                lista.append(r)
                lista.append(n0)
        r, n0 = lista
        return [t, K*np.exp(np.log(n0/K)*np.exp(-r*t))]


class Modelo_Logistico_Generalizado(Parametros):
        
    def logistico_generalizado(self, ID:str):
        K = 2000
        theta = 0.5
        t = np.linspace(0,28, 100)
        lista = []
        for i in self.dataset.index:
            id, _, r, n0, _ = self.dataset.loc[i].values
            if id == ID:
                lista.append(r)
                lista.append(n0)
        r, n0 = lista

        return [t, K*(1 + ((K/n0)**theta - 1)*np.exp(-r*t))**(-1/theta)]


class Modelo_Gompertz_Modificado():
    def __init__(self, dataset:pd.DataFrame):
        self.dataset = dataset
        self.IDs = []
        self.Cs = []
        self.rs = []
        self.n0s = []
        self.Lambidas = []
        self.RMSEs = []

    def lista_de_parametros(self, inicial, final):
        IDs = self.IDs
        Cs = self.Cs
        rs = self.rs
        n0s = self.n0s
        Lambdas = self.Lambidas
        RMSEs = self.RMSEs
    
        for valor in range(len(self.dataset.loc[inicial:final])):
            ID, C, r, n0, Lambida, RMSE = self.dataset.loc[inicial:final].values[valor]
            IDs.append(ID)
            Cs.append(C)
            rs.append(r)
            n0s.append(n0)
            Lambdas.append(Lambida)
            RMSEs.append(RMSE)
        
        dic = {"ID":IDs, "C":Cs, "r":rs, "n0":n0s, "lambda":Lambdas, "RMSE":RMSEs}
        return dic
    
    def gompertz_modificado(self, ID:str):
        K = 2000
        t = np.linspace(0,28, 100)
        lista = []
        for i in self.dataset.index:
            id, _, r, n0, Lambda, _ = self.dataset.loc[i].values
            if id == ID:
                lista.append(r)
                lista.append(n0)
        r, n0 = lista
        return K*np.exp(-np.exp(r * (np.exp(1)/K) * (Lambda - t) + 1))

