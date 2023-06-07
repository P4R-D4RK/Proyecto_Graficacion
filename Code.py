import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cv2
import numpy as np

names = ['cup-1', 'device2-1', 'device3-1', 'device4-1', 'device8-1', 'device9-1', 'flatfish-1', 'jar-1', 'personal_car-1', 'ray-1']
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
        '1': ('Funciones', chainCodes),
        '2': ('Mostrar imágenes', showImages),
        '3': ('Opción 3', accion3),
        '4': ('Salir', salir)
    }

    generar_menu(opciones, '4')

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

def AF8_2D(F8):
    Af8 = []
    for i in range(0,len(F8)):
        if int(F8[i]) >= int(F8[i-1]):
            Af8.append(int(F8[i]) - int(F8[i-1])) 
        if int(F8[i]) < int(F8[i-1]):
            Af8.append((int(F8[i]) + 8) - (int(F8[i-1])))    
    AF8="".join(map(str, Af8))
    return AF8

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

def chainCodes():
    chainCodeNames = ['F8', 'F4', 'AF8', '3OT', 'VCC']
    matrix_chainCodes2D = [[],[],[],[],[]]
    for i in range(len(images)):
        matrix_chainCodes2D[0].append(F8_2D(images[i]))
        matrix_chainCodes2D[1].append(F4_2D(images[i]))

    for i in range(len(images)):
        matrix_chainCodes2D[2].append(AF8_2D(matrix_chainCodes2D[0][i]))
        matrix_chainCodes2D[3].append(OT3_2D(matrix_chainCodes2D[1][i]))
    
    for i in range(len(images)):
        matrix_chainCodes2D[4].append(VCC_2D(matrix_chainCodes2D[3][i]))

    i = 0
    for chainCode in matrix_chainCodes2D:
        print('\n--',chainCodeNames[i],'--')
        for j in range(len(chainCode)):
            print(names[j],': ',chainCode[j])
        i += 1

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