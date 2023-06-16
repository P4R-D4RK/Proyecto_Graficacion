import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pyvista as pv
from PIL import Image
from skimage import measure
from plyfile import PlyData
import ezdxf

names = ['cup-1', 'device8-1', 'flatfish-1', 'jar-1', 'personal_car-1']
images = []
models_3d = ['cow', 'elephant', 'hippo', 'pear', 'torus']
chainCodeNames = ['F8', 'F4', 'AF8', '3OT', 'VCC']

def menu_principal():
    while True:
        print("\n=== MENÚ PRINCIPAL ===")
        print("1. Funciones")
        print("2. Abrir imágenes")
        print("3. Guardar")
        print("0. Salir")
        
        opcion = input("Selecciona una opción: ")
        
        if opcion == "1":
            sub_menu1()
        elif opcion == "2":
            sub_menu2()
        elif opcion == "3":
            guardar()
        elif opcion == "0":
            break
        else:
            print("Opción no válida. Por favor, selecciona una opción válida.")


def sub_menu1():
    while True:
        print("\n=== Funciones ===")
        print("1. Códigos de cadena")
        print("2. Graficar cadenas")
        print("3. Nubes de puntos")
        print("4. Euler")
        print("0. Regresar")
        
        opcion = input("Selecciona una opción: ")
        
        if opcion == "1":
            codes = chainCodes()
            sub_menu1_1(codes)
        elif opcion == "2":
            codes = chainCodes()
            sub_menu1_2(codes)
        elif opcion == "3":
            sub_menu1_3()
        elif opcion == "4":
            sub_menu1_4()
        elif opcion == "0":
            break
        else:
            print("Opción no válida. Por favor, selecciona una opción válida.")

def sub_menu1_1(codes):
    while True:
        print("\n=== Códigos de cadena ===")
        print("1. F8")
        print("2. F4")
        print("3. AF8")
        print("4. 3OT")
        print("5. VCC")
        print("0. Regresar")
        
        opcion = input("Selecciona una opción: ")
        
        if opcion == "1":
            print('\n--',chainCodeNames[0],'--')
            for j in range(len(codes[0])):
                print(names[j],': ',codes[0][j])
            F8_3D()
        elif opcion == "2":
            print('\n--',chainCodeNames[1],'--')
            for j in range(len(codes[1])):
                print(names[j],': ',codes[1][j])
            F4_3D()
        elif opcion == "3":
            print('\n--',chainCodeNames[2],'--')
            for j in range(len(codes[2])):
                print(names[j],': ',codes[2][j])            
            VCC_3D()
        elif opcion == "4":
            print('\n--',chainCodeNames[3],'--')
            for j in range(len(codes[3])):
                print(names[j],': ',codes[3][j])
            OT3_3D()
        elif opcion == "5":
            print('\n--',chainCodeNames[4],'--')
            for j in range(len(codes[4])):
                print(names[j],': ',codes[4][j])
            AF8_3D()
        elif opcion == "0":
            break
        else:
            print("Opción no válida. Por favor, selecciona una opción válida.")

def sub_menu1_2(codes):
    while True:
        print("\n=== Graficar cadenas ===")
        print("1. 2D")
        print("2. 3D")
        print("0. Regresar")
        
        opcion = input("Selecciona una opción: ")
        
        if opcion == "1":
            graficar_2d(codes)
        elif opcion == "2":
            for name in models_3d:
                graficar_3d('images/' + name + '.ply', 'saves/' + name + '.dxf')
        elif opcion == "0":
            break
        else:
            print("Opción no válida. Por favor, selecciona una opción válida.")

def sub_menu1_3():
    while True:
        print("\n=== Nube de puntos ===")
        print("1. 2D")
        print("2. 3D")
        print("0. Regresar")
        
        opcion = input("Selecciona una opción: ")
        
        if opcion == "1":
            nubes_2d()
        elif opcion == "2":
            nubes_3d()
        elif opcion == "0":
            break
        else:
            print("Opción no válida. Por favor, selecciona una opción válida.")

def sub_menu1_4():
    while True:
        print("\n=== Euler ===")
        print("1. Característica de Euler")
        print("2. Número de hoyos o túneles")
        print("3. Número de 1-pixeles/1-voxeles")
        print("4. Número de tetra-pixeles")
        print("5. Número de vertices")
        print("6. Número de aristas")
        print("0. Regresar")
        
        opcion = input("Selecciona una opción: ")
        
        if opcion == "1":
            caract_euler()
        elif opcion == "2":
            num_hoyos()
        elif opcion == "3":
            num_1pixeles()
            num_1voxeles()
        elif opcion == "4":
            num_tetra()
        elif opcion == "5":
            num_vertices_aristas()
        elif opcion == "6":
            num_vertices_aristas()
        elif opcion == "0":
            break
        else:
            print("Opción no válida. Por favor, selecciona una opción válida.")

def sub_menu2():
    while True:
        print("\n=== Abrir imágenes ===")
        print("1. 2D")
        print("2. 3D")
        print("0. Regresar")
        
        opcion = input("Selecciona una opción: ")
        
        if opcion == "1":
            show2DImages()
        elif opcion == "2":
            show3DImages()
        elif opcion == "0":
            break
        else:
            print("Opción no válida. Por favor, selecciona una opción válida.")

def chainCodes():
    matrix_chainCodes2D = [[],[],[],[],[]]
    for i in range(len(images)):
        matrix_chainCodes2D[0].append(F8_2D(images[i]))
        matrix_chainCodes2D[1].append(F4_2D(images[i]))

    for i in range(len(images)):
        matrix_chainCodes2D[2].append(AF8_2D(matrix_chainCodes2D[0][i]))
        matrix_chainCodes2D[3].append(OT3_2D(matrix_chainCodes2D[1][i]))
    
    for i in range(len(images)):
        matrix_chainCodes2D[4].append(VCC_2D(matrix_chainCodes2D[3][i]))

    return matrix_chainCodes2D

def caract_euler():
    for name in names:
        # Cargar la imagen en escala de grises
        image = cv2.imread('images/'+name+'-original.png', 0)

        # Aplicar la característica de Euler a la imagen
        result_image = euler_image(image, iterations=10, step_size=0.1)

        # Mostrar la imagen original y la imagen resultante
        cv2.imshow('Original Image', image)
        cv2.imshow('Result Image', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
def euler_image(image, iterations, step_size):
    """
    Aplica la característica de Euler a una imagen 2D.

    Args:
        image: Matriz NumPy que representa la imagen 2D (escala de grises)
        iterations: Número de iteraciones a realizar
        step_size: Tamaño del paso para la aproximación de las derivadas parciales

    Returns:
        La imagen resultante después de aplicar la característica de Euler
    """
    result = image.copy().astype(np.float32)
    h, w = image.shape

    for _ in range(iterations):
        dx, dy = np.gradient(result)
        result += step_size * (np.abs(dx) + np.abs(dy))

    # Escalar los valores resultantes a [0, 255] y convertir a tipo uint8
    result = ((result - np.min(result)) / np.ptp(result) * 255).astype(np.uint8)

    return result

def num_hoyos():
    return

def num_1pixeles():
    print()
    for name in names:
        # Cargar la imagen en escala de grises
        image = cv2.imread('images/' + name + '.png', 0)

        # Aplicar umbralización para convertir la imagen en binaria
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # Encontrar los píxeles con valor 1
        white_pixels = np.where(binary_image == 255)

        # Obtener las coordenadas de los píxeles con valor 1
        white_pixels_coordinates = list(zip(white_pixels[1], white_pixels[0]))

        # Imprimir la cantidad de píxeles con valor 1 y sus coordenadas
        print(f"{name} - Cantidad de 1-píxeles: {len(white_pixels_coordinates)}")


def num_1voxeles():
    return

def num_tetra():
    for name in names:
        imagen = Image.open('images/' + name + '.png')
        ancho, alto = imagen.size

        contador_tetrapixeles = 1 if ancho == alto else 0 

        for x in range(0, ancho - 1, 2):
            for y in range(0, alto - 1, 2):
                pixel1 = imagen.getpixel((x, y))
                pixel2 = imagen.getpixel((x + 1, y))
                pixel3 = imagen.getpixel((x, y + 1))
                pixel4 = imagen.getpixel((x + 1, y + 1))
                
                if pixel1 == pixel2 == pixel3 == pixel4:
                    contador_tetrapixeles += 1
        
        print(contador_tetrapixeles)


def num_vertices_aristas():
    print()
    for name in names:
        
        # Cargar la imagen en escala de grises
        imagen = cv2.imread('images/' + name + '.png', cv2.IMREAD_GRAYSCALE)

        # Aplicar el operador Canny para detectar las aristas
        bordes = cv2.Canny(imagen, 100, 200)

        # Encontrar los contornos de los objetos
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contar el número de vértices
        contador_vertices = 0
        for contorno in contornos:
            epsilon = 0.02 * cv2.arcLength(contorno, True)
            aproximacion = cv2.approxPolyDP(contorno, epsilon, True)
            contador_vertices += len(aproximacion)

        print(name,'-',contador_vertices)
    

    ruta_imagen = "tiff/pear.tiff"
    imagen = Image.open(ruta_imagen)

    # Convertir la imagen a escala de grises si es necesario
    if imagen.mode != "L":
        imagen = imagen.convert("L")

    aristas, vertices = calcular_aristas_vertices(np.array(imagen))

    print("Número de aristas:", aristas)
    print("Número de vértices:", vertices)

def calcular_aristas_vertices(imagen_3d):
    # Obtener los contornos de la imagen binaria 3D
    contornos = measure.find_contours(imagen_3d, 0.5)

    # Contar el número de aristas
    contador_aristas = len(contornos)

    # Contar el número de vértices
    contador_vertices = sum([len(contorno) for contorno in contornos])

    return contador_aristas, contador_vertices
        
def guardar():
    print('Guardar!')

def graficar_2d(matrix_chainCodes2D):
    fig, axs = plt.subplots(len(matrix_chainCodes2D), sharex=True, figsize=(8, 6))

    for i, cadena in enumerate(matrix_chainCodes2D):
        axs[i].plot(cadena, marker='o')
        axs[i].set_ylabel(f'Cadena {i+1}')
        axs[i].grid(True)

    axs[-1].set_xlabel('Índice')

    plt.tight_layout()
    plt.show()

def graficar_3d(mesh_path,output_path):
        # Cargar el archivo PLY como un objeto PolyData
    mesh = pv.read(mesh_path)

    # Obtener las coordenadas de los vértices
    vertices = mesh.points[:, :2]

    # Crear una figura y un eje
    fig, ax = plt.subplots()

    # Graficar los vértices de la malla
    ax.scatter(vertices[:, 0], vertices[:, 1], color='black', s=1)

    # Obtener las aristas de la malla
    edges = mesh.lines

    # Graficar las aristas de la malla como segmentos de recta
    for edge in edges:
        start = vertices[edge[0]]
        end = vertices[edge[1]]
        ax.plot([start[0], end[0]], [start[1], end[1]], color='red', linewidth=0.5)

    # Configurar los límites del eje
    if not np.isnan(vertices[:, 0]).any() and not np.isnan(vertices[:, 1]).any():
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())

    # Configurar el aspecto de los ejes para que sean iguales
    ax.set_aspect('equal')

    # Guardar la figura en un archivo vectorial
    # plt.savefig(output_path, format='svg')

    # Guardar los puntos y las líneas en un archivo DXF
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Agregar los puntos al archivo DXF
    for point in vertices:
        msp.add_point(point, dxfattribs={'layer': 'Points'})

    # Agregar las líneas al archivo DXF
    for edge in edges:
        start = vertices[edge[0]]
        end = vertices[edge[1]]
        msp.add_line(start, end, dxfattribs={'layer': 'Lines'})

    doc.saveas(output_path[:-4] + '.dxf')

    # Mostrar la imagen con las cadenas graficadas
    plt.show()

def nubes_2d():
    for name in names:
        # Lee la imagen en color
        image = cv2.imread('images/' + name + '-original.png')

        # Convierte la imagen a escala de grises
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplica una umbralización a la imagen
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Encuentra los contornos en la imagen binaria
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Encuentra los puntos de quiebre en los contornos
        break_points = []
        for contour in contours:
            approx_curve = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            if len(approx_curve) > 4:  # Puedes ajustar este umbral según tus necesidades
                break_points.extend(approx_curve)

        # Dibuja los puntos de quiebre en una imagen en blanco
        result_image = np.zeros_like(image)
        for point in break_points:
            cv2.circle(image, tuple(point[0]), 5, (0, 255, 0), -1)

        # print(break_points)
        # Calcula el Error Cuadrático Integral (ISE)
        ise = np.sum((image.astype("float") - result_image.astype("float")) ** 2)

        # Muestra el ISE
        print("ISE:", ise)

        # Muestra la imagen con los puntos de quiebre
        cv2.imshow('Break Points', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def nubes_3d():
    nube_puntos_3D = []
    
    for name in models_3d:
        nube_puntos_3D = []
        plydata = PlyData.read('images/'+name+'.ply')
        vertices = plydata['vertex'].data[['x', 'y', 'z']]
        
        nube_puntos_3D.append(vertices)
                
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for puntos in nube_puntos_3D:
            ax.scatter(puntos['x'], puntos['y'], puntos['z'], s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        doc = ezdxf.new("R2010")  # Crea un nuevo documento DWG en formato R2010

        msp = doc.modelspace()  # Obtiene el espacio de modelos

        for punto in vertices:
            msp.add_point(punto, dxfattribs={'layer': 'Points'})  # Agrega un punto en el espacio de modelos

        doc.saveas('saves/' + name + '.dwg')  # Guarda el archivo DWG

        print(f"Archivo DWG {name}.dwg exportado en carpeta saves.")


def F8_2D(image):
    F8 = ''
    i, j = 0, 0
    iy, ix, = 0, 0
    f = False

    for i in range(50):
        if(f):
            break
        for j in range(len(image[0])):
            if(image[i][j]==1):
                iy, ix = i, j
                f = True
                break

    i -= 1
    k, l = 0, 0
    f = False
    terminar = False
    izq = True
    z = 0

    while(terminar == False):
        if(z>0):
            if(i == iy and j == ix):
                terminar = True
                break
        
        f = False
        #print('Z',z)
        #print('i',i,'j',j)
        #print(izq)
        #print('i',i,'j',j)
        if(izq):
            for m in range(-1,2):
                if(i-1==k and j+m==l):
                    continue
                else:
                    if(image[i-1][j+m]==1):
                        if(m==-1): 
                            izq = False
                            F8 += '5'
                        if(m==0): F8 += '6'
                        if(m==1): F8 += '7'
                        # print(-1,m) 
                        k, l = i, j
                        i, j = i-1,j+m
                        f = True
                        break
                
            if(i==k and j+1==l):
                continue
            else:
                if(image[i][j+1]==1 and f==False):
                    F8 += '0'
                    # print(0,1) 
                    k, l = i, j
                    i, j = i,j+1
                    f = True
            
            if(f == False):
                    for m in range(-1,2):
                        if(i+1==k and j-m==l):
                            continue
                        else:
                            if(image[i+1][j-m]==1):
                                if(m==-1): F8 += '1'
                                if(m==0): 
                                    F8 += '2'
                                    izq = False
                                if(m==1): 
                                    F8 += '3'
                                    izq = False
                                # print(1,-m) 
                                k, l = i, j
                                i, j = i+1,j-m
                                f = True
                                break
                
            if(i==k and j-1==l):
                continue
            else:
                if(image[i][j-1]==1 and f==False):
                    F8 += '4'
                    # print(0,-1) 
                    k, l = i, j
                    i, j = i,j-1
                    izq = False
                    f = True
            
            if(f == False):
                temp1, temp2 = k, l
                k, l = i, j
                i, j = k, l
                
            
        
                    
        else:
            if(f == False):
                for m in range(-1,2):
                    if(i+1==k and j-m==l):
                        continue
                    else:
                        if(image[i+1][j-m]==1):
                            if(m==-1): 
                                izq = True
                                F8 += '1'
                            if(m==0): F8 += '2'
                            if(m==1): 
                                F8 += '3'
                            # print(1,-m) 
                            k, l = i, j
                            i, j = i+1,j-m
                            f = True
                            break
            
            if(i==k and j-1==l):
                continue
            else:
                if(image[i][j-1]==1 and f==False):
                    F8 += '4'
                    # print(0,-1) 
                    k, l = i, j
                    i, j = i,j-1
                    f = True

            if(f == False):          
                for m in range(-1,2):
                    if(i-1==k and j+m==l):
                        continue
                    else:
                        if(image[i-1][j+m]==1):
                            if(m==-1): 
                                F8 += '5'
                            if(m==0): 
                                F8 += '6'
                                izq = True
                            if(m==1): 
                                izq = True
                                F8 += '7'
                            # print(-1,m) 
                            k, l = i, j
                            i, j = i-1,j+m
                            f = True
                            break
            
            if(i==k and j+1==l):
                continue
            else:
                if(image[i][j+1]==1 and f==False):
                    F8 += '0'
                    # print(0,1) 
                    k, l = i, j
                    i, j = i,j+1
                    izq = True
                    f = True 
            
        z += 1

    return F8

def F8_3D():
    return

def F4_2D(image):
    F4 = ''
    i, j = 0, 0
    iy, ix, = 0, 0
    f = False

    for i in range(50):
        if(f):
            break
        for j in range(len(image[0])):
            if(image[i][j]==1):
                iy, ix = i, j
                f = True
                break

    i -= 1
    k, l = 0, 0
    f = False
    terminar = False
    izq = True
    z = 0

    while(terminar == False):
        if(z>0):
            if(i == iy and j == ix):
                terminar = True
                break
        
        f = False
        #print('Z',z)
        #print('i',i,'j',j)
        #print(izq)
        #print('i',i,'j',j)
        if(izq):

            for m in range(-1,2):
                if(i-1==k and j+m==l):
                    continue
                else:
                    if(image[i-1][j+m]==1):
                        if(m==-1): 
                            izq = False
                            F4 += '32' #32  Arriba y luego izquierda 
                        if(m==0): F4 += '3' #3 Arriba 
                        if(m==1): F4 += '03' #03 Derecha arriba 
                        # print(-1,m) 
                        k, l = i, j
                        i, j = i-1,j+m
                        f = True
                        break
                
            if(i==k and j+1==l):
                continue
            else:
                if(image[i][j+1]==1 and f==False):
                    F4 += '0' #Derecha 
                    # print(0,1) 
                    k, l = i, j
                    i, j = i,j+1
                    f = True
            
            if(f == False):
                    for m in range(-1,2):
                        if(i+1==k and j-m==l):
                            continue
                        else:
                            if(image[i+1][j-m]==1):
                                if(m==-1): 
                                    F4 += '10' #10 Abajo derecha 
                                if(m==0): 
                                    F4 += '1' #Abajo 
                                    izq = False
                                if(m==1): 
                                    F4 += '21' #Izquierda abajo 
                                    izq = False
                                # print(1,-m) 
                                k, l = i, j
                                i, j = i+1,j-m
                                f = True
                                break
                
            if(i==k and j-1==l):
                continue
            else:
                if(image[i][j-1]==1 and f==False):
                    F4 += '2' #Izquierda 
                    # print(0,-1) 
                    k, l = i, j
                    i, j = i,j-1
                    izq = False
                    f = True
            
            if(f == False):
                temp1, temp2 = k, l
                k, l = i, j
                i, j = k, l
                
            
        
                    
        else:
            if(f == False):
                for m in range(-1,2):
                    if(i+1==k and j-m==l):
                        continue
                    else:
                        if(image[i+1][j-m]==1):
                            if(m==-1): 
                                izq = True
                                F4 += '10' #01 Derecha abajo 
                            if(m==0): F4 += '1' #1 abajo 
                            if(m==1): 
                                F4 += '21' #12  abajo izquierda 
                            # print(1,-m) 
                            k, l = i, j
                            i, j = i+1,j-m
                            f = True
                            break
            
            if(i==k and j-1==l):
                continue
            else:
                if(image[i][j-1]==1 and f==False):
                    F4 += '2' #2 izquierda 
                    # print(0,-1) 
                    k, l = i, j
                    i, j = i,j-1
                    f = True

            if(f == False):          
                for m in range(-1,2):
                    if(i-1==k and j+m==l):
                        continue
                    else:
                        if(image[i-1][j+m]==1):
                            if(m==-1): 
                                F4 += '32' #23 izquierda arriba 
                            if(m==0): 
                                F4 += '3' #3 arriba 
                                izq = True
                            if(m==1): 
                                izq = True
                                F4 += '03' #30 arriba derecha 
                            # print(-1,m) 
                            k, l = i, j
                            i, j = i-1,j+m
                            f = True
                            break
            
            if(i==k and j+1==l):
                continue
            else:
                if(image[i][j+1]==1 and f==False):
                    F4 += '0' #0 derecha 
                    # print(0,1) 
                    k, l = i, j
                    i, j = i,j+1
                    izq = True
                    f = True 
            
        z += 1
        
    return F4

def F4_3D():
    return

def AF8_2D(F8):
    Af8 = []
    for i in range(0,len(F8)):
        if int(F8[i]) >= int(F8[i-1]):
            Af8.append(int(F8[i]) - int(F8[i-1])) 
        if int(F8[i]) < int(F8[i-1]):
            Af8.append((int(F8[i]) + 8) - (int(F8[i-1])))    
    AF8="".join(map(str, Af8))
    return AF8

def AF8_3D():
    return

def OT3_2D(F4):
    referencia= 3
    ot3=[]
    b = True
    for i in range(1,len(F4) ):
        if int(F4[i-1])==int(F4[i]):
            ot3.append(0)
        else:
            if int(F4[i])==int(referencia):
                referencia= F4[i-1]
                ot3.append(1)
                if (b == True): b=False 
                else: b = True 
            else:
                    referencia = F4[i-1]
                    if b : 
                        ot3.append(0)
                        ot3.append(2)
                    else:
                        
                        if int(F4[i-1])-int(F4[i])==2 or int(F4[i-1])-int(F4[i])==-2 :
                            ot3.append(1)
                            if (b == True): b=False 
                            else: b = True 
                            ot3.append(2)
                            referencia = int(F4[i-1])+1
                        else: 
                            ot3[-1]=2
    if  int(F4[-1])-int(F4[0])==2 or int(F4[-1])-int(F4[0])==-2:
        ot3.append(1)
        ot3.append(2)
    elif int(F4[0]) == int(referencia):
        ot3.append(1)
    else:
        ot3.append(2)

    OT3 = "".join(map(str, ot3))
    return OT3

def OT3_3D():
    return

def VCC_2D(OT3):
    VCC=""
    bo='1'
    for i in range(len(OT3)):
        if OT3[i]=='0':VCC += '0'
        if OT3[i]=='1':
            if bo=='1':bo='2'
            elif bo=='2':bo='1'
            VCC += bo
        if OT3[i]=='2':VCC+= bo
    return VCC

def VCC_3D():
    return


def show2DImages():
    fig,axs = plt.subplots(1,5,figsize=(10,5))
    # Mostrar cada imagen en un subplot
    axs[0].imshow(images[0], cmap='gray')
    axs[0].axis('off')
    axs[0].set_title(names[0])

    axs[1].imshow(images[1], cmap='gray')
    axs[1].axis('off')
    axs[1].set_title(names[1])

    axs[2].imshow(images[2], cmap='gray')
    axs[2].axis('off')
    axs[2].set_title(names[2])

    axs[3].imshow(images[3], cmap='gray')
    axs[3].axis('off')
    axs[3].set_title(names[3])

    axs[4].imshow(images[4], cmap='gray')
    axs[4].axis('off')
    axs[4].set_title(names[4])

    # Ajustar los espacios entre subplots
    plt.tight_layout()
    plt.show()

def show3DImages():
    plotter = pv.Plotter(shape=(1, 5),window_size=[800,300], )

    malla = []
    for img in models_3d:
        malla.append(pv.read('images/'+img+'.ply'))

    plotter.subplot(0, 0)
    plotter.add_text(models_3d[0], font_size=15)
    plotter.add_mesh(malla[0])

    plotter.subplot(0, 1)
    plotter.add_text(models_3d[1], font_size=15)
    plotter.add_mesh(malla[1])

    plotter.subplot(0, 2)
    plotter.add_text(models_3d[2], font_size=15)
    plotter.add_mesh(malla[2])

    plotter.subplot(0, 3)
    plotter.add_text(models_3d[3], font_size=15)
    plotter.add_mesh(malla[3])

    plotter.subplot(0, 4)
    plotter.add_text(models_3d[4], font_size=15)
    plotter.add_mesh(malla[4])

    # Display the window
    plotter.show()

if __name__ == '__main__':
    for name in names:
        fullName = 'images/' + name + '.png'
        img = np.array(cv2.imread(fullName, cv2.IMREAD_GRAYSCALE))/255
        images.append(img.astype(int))
    menu_principal()