#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:10:46 2022

@author: miguel
"""

from knn import KNN
from eigenfaces import EIGENFACES
import numpy as np
import imagenes
import statistics

# ---------------     Carga de datos     --------------------
# Función que carga en memoria el conjunto de datos de caras fei
# el conjunto debe estar etiquetado correctamente. Si el nombre de la imagen contiene h, el algoritmo lo identifica como hombre
# si el nombre de la imagen contiene m, el algoritmo lo identifica como mujer. Ej. 1ah,jpg -> hombre, 1bm.jpg -> mujer
def load_fei(n_test):
    dataset_path = "fei/frontalimages_manuallyaligned_part1/"
    n_training = 400 - n_test
    print("# datos de entrenamiento:", n_training, ", # de datos prueba:", n_test)
    # Importa datos
    train_attrib, train_labels, test_attrib, test_labels = imagenes.load_data(
        dataset_path, n_training
    )
    return train_attrib, train_labels, test_attrib, test_labels, n_test

def load_fei_2(n_test):
    dataset_path = "fei/frontalimages_manuallyaligned_part1/"
    n_training = 400 - n_test
    print("# datos de entrenamiento:", n_training, ", # de datos prueba:", n_test)
    # Importa datos
    train_attrib, train_labels, test_attrib, test_labels = imagenes.load_data_2(
        dataset_path, n_training
    )
    return train_attrib, train_labels, test_attrib, test_labels, n_test

# Ejecuta KNN para clasificacion usando el conjunto de prueba test_attrib, test_labels, n_testing
def knn_classic_clasify(clasificador, k, test_attrib, test_labels, n_testing):
    good = 0
    bad = 0
    #print(train_attrib, train_labels)
    for a in range(len(test_attrib[0])):
        atrib = []
        for i in range(len(test_attrib)):
            atrib.append(test_attrib[i][a])
        #print("prueba: ", a, " Data:", atrib)
        
        res = clasificador.clasificar(atrib, k)
        
        index, name = test_labels[a]
        
        if (name == res[0]):
            good = good + 1
            #print("Esperado:", name, " Obtenido:", res, "OK")
        else:
            bad = bad +1
            #print("Esperado:", name, " Obtenido:", res, "FALLÓ")
            
    good_p = (good/n_testing)*100
    bad_p = (bad/n_testing)*100
    print("Good(",good,"): ", good_p, "%, Bad(",bad,"):", bad_p, "%, total: ", n_testing)
    return [good_p, bad_p]


# Ejecuta KNN para clasificacion usando el conjunto de prueba test_attrib, test_labels, n_testing
def knn_clasify(eigenfaces, clasificador, k, test_attrib, test_labels, n_testing):
    good = 0
    bad = 0
    #print(train_attrib, train_labels)
    for a in range(len(test_attrib)):
        #atrib = []
        #for i in range(len(test_attrib)):
        #    atrib.append(test_attrib[i][a])
        #print("prueba: ", a, " Data:", atrib)
        #eigenfaces.show_face(test_attrib[a])
        atrib = eigenfaces.calc_pesos_knn(test_attrib[a])
        
        #atrib = eigenfaces.normalize(atrib)
        #print(test_labels[a])
        #print(atrib)
        #exit()
        res = clasificador.clasificar(atrib, k)
        
        index, name = test_labels[a]
        
        if (name == res[0]):
            good = good + 1
            #print("Esperado:", name, " Obtenido:", res, "OK")
        else:
            bad = bad +1
            #print("Esperado:", name, " Obtenido:", res, "FALLÓ")
        #exit()
    good_p = (good/n_testing)*100
    bad_p = (bad/n_testing)*100
    print("Good(",good,"): ", good_p, "%, Bad(",bad,"):", bad_p, "%, total: ", n_testing)
    return [good_p, bad_p]

# Ejecuta EIgenfaces para clasificacion usando el conjunto de prueba test_attrib, test_labels, n_testing
def eigen_clasify(eigenfaces, test_attrib, test_labels, n_testing):
    good = 0
    bad = 0
    #print(train_attrib, train_labels)
    for a in range(len(test_attrib)):
        
        res = eigenfaces.recognize(test_attrib[a])
        
        index, name = test_labels[a]
        
        if (name == res[0]):
            good = good + 1
            #print("Esperado:", name, " Obtenido:", res, "OK")
        else:
            bad = bad + 1
            #print("Esperado:", name, " Obtenido:", res, "FALLÓ")
            
    good_p = (good/n_testing)*100
    bad_p = (bad/n_testing)*100
    print("Good(",good,"): ", good_p, "%, Bad(",bad,"):", bad_p, "%, total: ", n_testing)
    return [good_p, bad_p]

def ejecuta_clasificador_knn_fei(n_training):
    # Importa datos
    train_attrib, train_labels, test_attrib, test_labels, n_testing = load_fei(n_training)
    
    eigenfaces = EIGENFACES()
    eigenfaces.load_data(train_attrib, train_labels, True, 0.98)
    
    clasificador = KNN()
    #Calcula medias, desviacion estandar, etc.
    #clasificador.load_data(train_attrib, train_labels)
    #print(eigenfaces.w, eigenfaces.labels)
    clasificador.load_data(eigenfaces.w, eigenfaces.labels)
    
    k=31
    
    return knn_clasify(eigenfaces, clasificador, k, test_attrib, test_labels, n_testing)

def ejecuta_clasificador_eigenfaces(n_training):
    # Importa datos
    train_attrib, train_labels, test_attrib, test_labels, n_testing = load_fei(n_training)
    
    eigenfaces = EIGENFACES()
    eigenfaces.load_data(train_attrib, train_labels, False, 0.98)
    
    
    return eigen_clasify(eigenfaces, test_attrib, test_labels, n_testing)

def ejecuta_clasificador_knn_classic_fei(n_training):
    # Importa datos
    train_attrib, train_labels, test_attrib, test_labels, n_testing = load_fei_2(n_training)
    
    clasificador = KNN()
    #Calcula medias, desviacion estandar, etc.
    clasificador.load_data(train_attrib, train_labels)
    
    k=7
    
    return knn_classic_clasify(clasificador, k, test_attrib, test_labels, n_testing)
    

# Funcion principal
if __name__ == '__main__':
    n_iter = 10
    e_good = []
    e_bad = []
    for i in range(n_iter):
        
        # Define la longitud del conjunto de prueba
        n_test =100
        
        # Seleccion de prueba a ejecutar
        
        # Clasificación:
        good, bad = ejecuta_clasificador_knn_fei(n_test)
        #good, bad = ejecuta_clasificador_knn_classic_fei(n_test)
        #good, bad = ejecuta_clasificador_eigenfaces(n_test)
            
        e_good.append(good)
        e_bad.append(bad)
    
    prom_e_good = round(statistics.mean(e_good),2)
    prom_e_bad = round(statistics.mean(e_bad),2)
    
    dev_std_good = round(statistics.stdev(e_good),2)
    dev_std_bad =  round(statistics.stdev(e_bad),2)
    
    print("promedio Good: ", prom_e_good, "% , promedio bad: ", prom_e_bad, "%, # de iteraciones: ", n_iter)
    print("dev std: ", dev_std_good, "% , promedio bad: ", dev_std_bad, "%")