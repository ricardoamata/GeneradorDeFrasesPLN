import kenlm
import nltk
import random
import re
from util import limpia_donquijote, segmenta_texto, dataset_split

def main():
    model = kenlm.Model("quijote_lm.binary")

    texto_prueba = ""
    with open("datos/texto_prueba.txt", 'r', encoding='utf8') as file:
        texto_prueba = file.read()

    texto_entrenamiento = ""
    with open("datos/texto_entrenamiento.txt", 'r', encoding='utf8') as file:
        texto_entrenamiento = file.read()
    
    num_lineas = len(texto_prueba.split("\n"))

    p = 0
    for linea in texto_prueba.split("\n"):
        p += model.perplexity(linea)
    print(f"La perplejidad es: {p/num_lineas}")

    palabras = nltk.word_tokenize(texto_entrenamiento)
    palabras = set(palabras)
    palabras = list(palabras)
    numero_palabras = 30

    n = 1300
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

    print(f"\n{oracion}\n")

if __name__ == "__main__":
    main()

    