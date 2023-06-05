import cv2

names = ['apple-1', 'children-1', 'cup-1', 'device3-1', 'device4-1', 'Heart-1', 'personal_car-1', 'ray-1', 'sea_snake-1', 'turtle-1']

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
        '2': ('Mostrar imágenes', accion2),
        '3': ('Opción 3', accion3),
        '4': ('Salir', salir)
    }

    generar_menu(opciones, '4')


def accion1():
    print('Has elegido la opción 1')


def accion2():
    for name in names:
        fullName = 'images/' + name + '.png'
        image = cv2.imread(fullName) 
        cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def accion3():
    print('Has elegido la opción 3')


def salir():
    print('Saliendo')


if __name__ == '__main__':
    menu_principal()