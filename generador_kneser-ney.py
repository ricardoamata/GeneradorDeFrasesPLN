import kenlm
from nltk import ngrams, sent_tokenize, word_tokenize
from nltk.lm.counter import NgramCounter
import random
import re
from util import limpia_donquijote, segmenta_texto, dataset_split

def main():
    model = kenlm.Model("quijote_lm.binary")

    texto = ""
    with open("datos/DonQuijote.txt", 'rt', encoding='utf8') as file:
        texto = file.read()

    texto_prueba = ""
    with open("datos/texto_prueba.txt", 'r', encoding='utf8') as file:
        texto_prueba = file.read()

    texto_entrenamiento = ""
    with open("datos/texto_entrenamiento.txt", 'r', encoding='utf8') as file:
        texto_entrenamiento = file.read()

    texto = limpia_donquijote(texto)
    texto_tokenizado = [list(map(str.lower, word_tokenize(oracion)))
                      for oracion in sent_tokenize(texto)]
    
    vocabulario = len(set([palabra for oracion in texto_tokenizado for palabra in oracion]))
    print(f"El vocabulario del texto es de {vocabulario} palabras")

    fourgrams = [ngrams(oracion, 5) for oracion in texto_tokenizado]
    counter = NgramCounter(fourgrams)
    print(f"El texto contiene {counter.N()} 5-gramas")

    p = model.perplexity(texto_prueba)
    print(f"La perplejidad es: {p}")
    
    palabras = word_tokenize(texto_entrenamiento)
    palabras = set(palabras)
    palabras = list(palabras)
    print(f"El vocavulario del training set es de {len(palabras)} palabras")
    numero_palabras = 30

    n = 1000
    oracion = random.choice(palabras)
    for _ in range(numero_palabras):
        mejor_s = None
        mejor_p = None
        for palabra in random.choices(palabras, k=n):
            if mejor_s == None:
                mejor_s = model.score(oracion+" "+palabra)
                mejor_p = palabra
            elif mejor_s < model.score(oracion+" "+palabra):
                mejor_s = model.score(oracion+" "+palabra)
                mejor_p = palabra
        oracion = oracion + " " + mejor_p

    oracion = re.sub("(\\.) \\1", "\\1", oracion)
    oracion = re.sub(r"([,;:\.])", "\b\\1", oracion)
    print(f"\n{oracion}\n")

if __name__ == "__main__":
    main()

    