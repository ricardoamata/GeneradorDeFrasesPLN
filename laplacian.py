import re
from random import random
from nltk import download, compat, sent_tokenize, word_tokenize, ngrams
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.api import LanguageModel
from nltk.lm.counter import NgramCounter
from util import limpia_donquijote, segmenta_texto, dataset_split
download('punkt')

@compat.python_2_unicode_compatible
class MLE(LanguageModel):
    """Class for providing MLE ngram model scores.
    Inherits initialization from BaseNgramModel.
    """

    def unmasked_score(self, word, context=None):
        """Add-one smoothing: Laplace.
        """
        counts = self.context_counts(context)
        word_count = counts[word]
        norm_count = counts.N()
        return (word_count + 1) / (norm_count + len(self.vocab) + 1)

if __name__ == "__main__":
    n = 2

    # Cargamos Don Quijote
    text = ""
    with open("datos/DonQuijote.txt", 'rt', encoding='utf8') as file:
        text = file.read()
    # Limpiamos el tecto
    text = limpia_donquijote(text)
    # Lo tokenizamos
    texto_tokenizado = [list(map(str.lower, word_tokenize(oracion)))
                      for oracion in sent_tokenize(text)]
    # Obtenemos el vocabulario de todo el texto
    vocabulario = len(set([palabra for oracion in texto_tokenizado for palabra in oracion]))
    print(f"El vocabulario del texto es de {vocabulario} palabras")
    # Contamos el numero de ngramas que tiene
    grams = [ngrams(oracion, n) for oracion in texto_tokenizado]
    counter = NgramCounter(grams)
    print(f"El texto contiene {counter.N()} {n}-gramas")
    
    # Cargamos el texto de prueba
    texto_prueba = ""
    with open("datos/texto_prueba.txt", 'r', encoding='utf8') as file:
        texto_prueba = file.read()
    # Lo tokenizamos
    texto_prueba_tokenizado = [list(map(str.lower, word_tokenize(oracion)))
                      for oracion in sent_tokenize(texto_prueba)]
    # Obtenenmos los ngramas de prueba
    testgrams = list(ngrams(texto_prueba_tokenizado, n))

    # Cargamos el texto de entrenamiento
    texto_entrenamiento = ""
    with open("datos/texto_entrenamiento.txt", 'r', encoding='utf8') as file:
        texto_entrenamiento = file.read()
    # Lo tokenizamos
    texto_entrenamiento_tokenizado = [list(map(str.lower, word_tokenize(oracion)))
                      for oracion in sent_tokenize(texto_entrenamiento)]
    # Obtenenemos el vocabulario de el texto de entrenamiento
    train_v = list(set([palabra for oracion in texto_entrenamiento_tokenizado for palabra in oracion]))
    # Obtenemos los ngramas de entrenamiento
    traingrams = [ngram for oracion in texto_entrenamiento_tokenizado for ngram in ngrams(oracion, n)]
    
    datos_entrenamiento, sentencias_con_etiquetas = padded_everygram_pipeline(n, texto_entrenamiento_tokenizado)

    model = MLE(n)
    model.fit(datos_entrenamiento, sentencias_con_etiquetas)
    print("\nEntrenamiento terminado\n")
    p = model.perplexity(texto_prueba_tokenizado)
    print(f"La perplejidad es: {p}")
    texto_generado = model.generate(20, None, random())
    print(' '.join(texto_generado))