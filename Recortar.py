import cv2
import numpy as np
import os

# font = cv2.FONT_HERSHEY_COMPLEX
def recortar(img):
    width = 30
    height = 50
    def find_redctangles(conts, img):
        # Ordena as maiores areas para pegar somente os caracteres. São só os primeiros 7 pq são 7 caracteres
        conts = sorted(conts, key=cv2.contourArea, reverse=True)[:7]
        rects = []
        for c in conts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            rect = (x, y, w, h)
            rects.append(rect)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return (img, rects)

    def crop_rectangles(img, rect):
        # Inicia um vetor vazio
        vec = []
        for i in range(7):
            # Pega os retangulos e seleciona somente as letras separadas
            aux = img[rect[i][1]:rect[i][1]+rect[i][3], rect[i][0]:rect[i][0]+rect[i][2]]
            vec.append(aux)
        return vec


    # Faz o threshold da imagem
    _, threshold = cv2.threshold(img, 220, 255, cv2.THRESH_OTSU)

    # Faz a inversão da imagem
    threshold = 255 - threshold

    # Encontra os contornos da imagem
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Pinta os contornos da imagem
    img = cv2.drawContours(cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR),contours,-1, (0,255,0), 3)

    # Encontra os retangulos ao redor dos caracteres e os marca
    image_and_rectangles = find_redctangles(contours, threshold)
    rectangles = image_and_rectangles[1]

    # Ordena para pegar do menor valor de X para o maior
    rectangles.sort()

    # Recorta os caracteres e coloca em um vetor
    vec_final = crop_rectangles(threshold, rectangles)

    dim = (width, height)

    # for i in range(len(vec_final)):
    #     vec_final[i] = cv2.resize(vec_final[i] , dim, interpolation= cv2.INTER_AREA)
    # for i in range(len(vec_final)):
    #     vec_final[i] = cv2.resize(vec_final[i] , (1500, 1))
    # for i in range(7):
    #     vec_final[i] = vec_final[i] / 255
    #     vec_final[i] = vec_final[i][0]
    # vec_final = np.asfarray(vec_final)



    for i in range(len(vec_final)):
        # vec_final[i] = cv2.cvtColor(vec_final[i], cv2.COLOR_BGR2GRAY)
        trh, vec_final[i] = cv2.threshold(vec_final[i], 128, 255, cv2.THRESH_OTSU)
        vec_final[i] = cv2.resize(vec_final[i], dim)
        #salvar
        directory = r'/Users/gustavozagocanal/Desktop/Coisas trabalho ia/Imagens'
        os.chdir(directory)
        filename = 'teste_' + str(i) + '.jpg'
        cv2.imwrite(filename, vec_final[i])

        vec_final[i] = cv2.resize(vec_final[i], (1500, 1))
        vec_final[i] = vec_final[i] / 255
        vec_final[i] = vec_final[i][0]
    vec_final = np.asfarray(vec_final)

    # cv2.imshow("vec_0", vec_final[0])
    # cv2.imshow("vec_1", vec_final[1])
    # cv2.imshow("vec_2", vec_final[2])
    # cv2.imshow("vec_3", vec_final[3])
    # cv2.imshow("vec_4", vec_final[4])
    # cv2.imshow("vec_5", vec_final[5])
    # cv2.imshow("vec_6", vec_final[6])

    # Salvar as imagens
    # for i in range(len(vec_final)):
    #     directory = r'/Users/gustavozagocanal/Desktop/Coisas trabalho ia/Imagens'
    #     os.chdir(directory)
    #     filename = 'teste_' + i + '.jpg'
    #     cv2.imwrite(filename, vec_final[i])



    # cv2.imshow("shapes", img)
    # cv2.imshow("Threshold", threshold)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return vec_final