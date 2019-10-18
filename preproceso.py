import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import download, Text
from nltk.util import ngrams
from random import random
from collections import Counter
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
download('punkt')

def limpia_donquijote(text):
    # Ponemos un token donde inicia el texto
    text = re.sub(r"Capítulo primero\. .*\n.*\n", "<START>", text)
    # Eliminamos todo lo que este antes de este token
    text = re.sub(r"^(.*\n)*<START>", "<START>", text)
    # Ponemos un token al final del texto
    text = re.sub(r"Fin\n", "<END>", text)
    # Eliminamos todo lo que este despues
    text = re.sub(r"\n<E>(.*\n)*$", "<END>", text)
    # Removemos el titulo de cada capitulo
    text = re.sub(r"(\n){5}Capítulo .*\.(.*\n){3}", "", text)
    # Si hay mas de 3 espacios los eliminamos
    text = re.sub(r"(\n){3,100}", "\n\n", text)
    # Removemos los caracteres inecesarios
    text = re.sub("[\'\"«»¿?!¡-]", "", text)
    # Removemos los caracteres inecesarios

    return text

def ngramaFecuencias(ngramas):
    """frecuencias = {}
    for ngrama in ngramas:
        if ngrama in frecuencias:
            frecuencias[ngrama] += 1
        else:
            frecuencias[ngrama] = 1"""
    return Counter(ngramas)

def oracionVerosimilitudLambda(oracion, n, lambdaC = 1):
    def probabilidad():
        pass

    pass

if __name__ == "__main__":
    text = ""
    with open("datos/DonQuijote.txt", 'rt', \
              encoding='utf8') as file:
        text = file.read()

    text = limpia_donquijote(text)

    texto_tokenizado = [list(map(str.lower, word_tokenize(oracion)))
                      for oracion in sent_tokenize(text)]
    for i, key in enumerate(texto_tokenizado):
        if i == 0:
            break
        print(key)
    #frecuencias = ngramaFecuencias(ngrams(words, 4))

    #for i, (key, value) in enumerate(frecuencias.items()):
    #    if i == 25:
    #        break
    #    print(key, value)
    n = 4
    datos_entrenamiento, sentencias_con_etiquetas = padded_everygram_pipeline(n, texto_tokenizado)

    model = MLE(n)  # oracionVerosimilitud
    model.fit(datos_entrenamiento, sentencias_con_etiquetas)
    texto_generado = ""
    for elem in model.generate(25, None, random()):
        if elem != "</s>":
            texto_generado += " " + elem
    print(texto_generado)
    
    with open("datos/output.txt", 'w') as file:
        file.write(text)




    
    
