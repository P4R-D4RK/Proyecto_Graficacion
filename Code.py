import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cv2
import numpy as np

names = ['apple-1', 'children-1', 'cup-1', 'device3-1', 'device4-1', 'Heart-1', 'personal_car-1', 'ray-1', 'sea_snake-1', 'turtle-1']
images = []

def mostrar_menu(opciones):
    print('Seleccione una opción:')
    for clave in sorted(opciones):
        print(f' {clave}) {opciones[clave][0]}')


def leer_opcion(opciones):
    while (a := input('Opción: ')) not in opciones:
        print('Opción incorrecta, vuelva a intentarlo.')
    return a


def ejecutar_opcion(opcion, opciones):
    opciones[opcion][1]()


def generar_menu(opciones, opcion_salida):
    opcion = None
    while opcion != opcion_salida:
        mostrar_menu(opciones)
        opcion = leer_opcion(opciones)
        ejecutar_opcion(opcion, opciones)
        print()


def menu_principal():
    opciones = {
        '1': ('Opción 1', accion1),
        '2': ('Mostrar imágenes', showImages),
        '3': ('Opción 3', accion3),
        '4': ('Salir', salir)
    }

    generar_menu(opciones, '4')


def accion1():
    print('Has elegido la opción 1')


def showImages():
    fig,axs = plt.subplots(2,5,figsize=(6,4))
    for i,img in enumerate(images):
        row = i//5
        col = i%5
        axs[row,col].imshow(img,cmap = "Greys_r")
        axs[row,col].axis("off")
    plt.show()


def accion3():
    print('Has elegido la opción 3')


def salir():
    print('Saliendo')


if __name__ == '__main__':
    for name in names:
        fullName = 'images/' + name + '.png'
        img = np.array(cv2.imread(fullName, cv2.IMREAD_GRAYSCALE))/255
        images.append(img.astype(int))
    menu_principal()