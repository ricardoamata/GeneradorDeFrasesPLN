import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import download
from random import random
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.api import LanguageModel
from nltk import compat
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
    text = ""
    with open("datos/DonQuijote.txt", 'rt', \
              encoding='utf8') as file:
        text = file.read()
    text = limpia_donquijote(text)
    texto_tokenizado = [list(map(str.lower, word_tokenize(oracion)))
                      for oracion in sent_tokenize(text)]
    n = 4
    datos_entrenamiento, sentencias_con_etiquetas = padded_everygram_pipeline(n, texto_tokenizado)

    model = MLE(n)  # oracionVerosimilitud
    model.fit(datos_entrenamiento, sentencias_con_etiquetas)
    texto_generado = model.generate(35, None, random())
    print(texto_generado)
