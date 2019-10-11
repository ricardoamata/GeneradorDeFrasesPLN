import re

def limpia_donquijote(text):
    # Ponemos un token donde inicia el texto
    text = re.sub(r"Capítulo primero\. .*\n.*\n", "<START>", text)
    # Eliminamos todo lo que este antes de este token
    text = re.sub(r"^(.*\n)*<START>", "<START>", text)
    # Ponemos un token al final del texto
    text = re.sub(r"Fin\n", "<END>", text)
    # Eliminamos todo lo que este despues
    text = re.sub(r"\n<END>(.*\n)*$", "<END>", text)
    # Removemos el titulo de cada capitulo
    text = re.sub(r"(\n){5}Capítulo .*\.(.*\n){3}", "", text)
    # Si hay mas de 3 espacios los eliminamos
    text = re.sub(r"(\n){3,100}", "\n\n", text)
    # Removemos los caracteres inecesarios
    text = re.sub("[\'\"«»¿?!¡-]", "", text)

    return text

if __name__ == "__main__":
    text = ""
    with open("datos/DonQuijote.txt", 'rt') as file:
        text = file.read()
    
    text = limpia_donquijote(text)

    with open("datos/output.txt", 'w') as file:
        file.write(text)


    
    