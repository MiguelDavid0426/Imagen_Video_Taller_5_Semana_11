import cv2 
import os 
import glob 
import sys
import numpy as np

#### Se ingresa la ruta donde estan las imagenes 
img_dir = "C:/Users/c72297a/OneDrive - EXPERIAN SERVICES CORP/Escritorio/imagenes_videos" 

### Se cargan las imagenes, se ajusta un tamaño 360x360 y se agraga a una base de datos "data"
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
data = [] 
for ima in files: 
    img = cv2.imread(ima) 
    img = cv2.resize(img, (360, 360))
    data.append(img) 
    
del data_path, files, ima, img, img_dir

##### Se muesta información de la base y se solicita la imagen de referencia para luego validar el valor
print("El número de imagenes en la ruta ingresada son:",len(data))
print("¿Cual de las",len(data),"imagenes, se usará como referencia?")
ref = int(input())-1
if ref in range(len(data)):
    None    
else:
    print(ref, "No es valido, por favor introducir un número entre 1 y", len(data))

#%%
### Se concatenan las imagenes para poder ingresar los puntos del usuario, intercalados uno en la imagen
### izquierda y el otro en la derecha, cambia color, finaliza con "x"
Tranformacion = []
for i in range(len(data)-1):
    imagen1 = data[i]
    imagen2 = data[i+1]
    image = cv2.hconcat([imagen1, imagen2])
    
    points = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
    
    image_draw = np.copy(image)
    
    points1 = []
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click)
    
    point_counter = 0
    c=0
    while True:
        if c%2 == 0:
            color = [0, 0, 255]
        else:
            color = [255, 0, 0]
            
        cv2.imshow("Image", image_draw)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            points1 = points.copy()
            points = []
            break
        if len(points) > point_counter:
            point_counter = len(points)
            cv2.circle(image_draw, (points[-1][0], points[-1][1]), 3, color, -1)
            c = c +1   
    points_1 = []
    points_2 = []
    for i in range(len(points1)):
        if i%2 == 0:
            points_1.append(points1[i])
        else:
            val = (points1[i][0]-image_draw.shape[0],points1[i][1])
            points_2.append(val)
    
    N = min(len(points_1), len(points_2))
    assert N >= 4, 'Se requieren minimo 4 punto por imagen'
    
    pts1 = np.array(points_1[:N])
    pts2 = np.array(points_2[:N])
    
    H, _ = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)
    
    Tranformacion.append(H)
#### Se separan los puntos en dos para calculas su tranformacion    
#%% 
#### Calcula las tranformaciones a la imagen de referencia, ya sea por debajo o opor encima
Trans_ima = []
for i in range(len(data)):
    TFF = np.identity(3)
    if i < ref:
        j = i
        while j<ref:
            TF = Tranformacion[j]
            TFF = np.dot(TFF, TF)
            j += 1
    elif i > ref:
        j = ref
        while j<i:
            TF = Tranformacion[j]
            TFF = np.dot(TFF, TF)
            j += 1   
        TFF = np.linalg.inv(TFF)

    image_warped = cv2.warpPerspective(data[i], TFF, (data[i].shape[1], data[i].shape[0]))
    Trans_ima.append(image_warped)
    
#### Multiplica la tranformacion (matriz 3x3) con su respectiva imagen y guarda las imagenes resultantes en Trans_ima
### Se debe promediar ls imagenes de image_warped y terminar el taller 
#%%      
avg_image = data[ref]
for i in range(len(Trans_ima)):
    if i == ref:
        pass
    else:
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        avg_image = cv2.addWeighted(Trans_ima[i], alpha, avg_image, beta, 0.0)

cv2.imshow("Image warped", avg_image)
cv2.waitKey(0)
        
#%%
# List of images, all must be the same size and data type.
avg_img = np.mean(Trans_ima, axis=0)
avg_img = avg_img.astype(np.uint8)
cv2.imshow("Image warped", avg_img)
cv2.waitKey(0)