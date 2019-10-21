from util import limpia_donquijote, segmenta_texto, dataset_split

if __name__ == "__main__":
    texto = ""
    with open("datos/DonQuijote.txt", 'rt', encoding='utf8') as file:
        texto = file.read()

    texto = limpia_donquijote(texto)

    texto_segmentado = segmenta_texto(texto)

    texto_entrenamiento, texto_prueba = dataset_split(texto_segmentado)
    
    with open('datos/texto_entrenamiento.txt', 'w', encoding='utf-8') as file:
        file.write(texto_entrenamiento)
    
    with open('datos/texto_prueba.txt', 'w', encoding='utf-8') as file:
        file.write(texto_prueba)





    
    
