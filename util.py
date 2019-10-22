import re
import nltk
import random

def limpia_donquijote(text):
    # Ponemos un token donde inicia el texto
    text = re.sub(r"Capítulo primero\. .*\n.*\n", "<START>", text)
    # Eliminamos todo lo que este antes de este token
    text = re.sub(r"^(.*\n)*<START>", "", text)
    # Ponemos un token al final del texto
    text = re.sub(r"Fin\n", "<END>", text)
    # Eliminamos todo lo que este despues
    text = re.sub(r"\n<END>(.*\n)*$", "", text)
    # Removemos el titulo de cada capitulo
    text = re.sub(r"(\n){5}Capítulo .*\.(.*\n){3}", "", text)
    # Si hay mas de 3 espacios los eliminamos
    text = re.sub(r"(\n){3,100}", "\n\n", text)
    # Removemos los caracteres inecesarios
    text = re.sub(r"[\'\"\(\)«»¿\?!¡-]", "", text)
    # Removemos los caracteres inecesarios

    return text

def segmenta_texto(texto):
    resultado = ""
    for oracion in nltk.sent_tokenize(texto):
        resultado += ' '.join(nltk.word_tokenize(oracion)).lower() + '\n'
    resultado = re.sub(r'[:;,\.]', '', resultado)
    return resultado

def dataset_split(data, train_size=0.8):
    dataset = data.split("\n")
    random.shuffle(dataset)

    train, test = dataset[:int(len(dataset)*train_size)], dataset[int(len(dataset)*train_size):]
    texto_entrenamiento = ""
    texto_prueba = ""

    for line in train:
        texto_entrenamiento += line +'\n'

    for line in test:
        texto_prueba += line + '\n'
    
    return texto_entrenamiento, texto_prueba