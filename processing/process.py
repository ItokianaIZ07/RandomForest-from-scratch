import cv2
import numpy as np
import pandas as pd

class DataProcessing:
    def __readImg(self, imgPath):
        return cv2.imread(imgPath)
    
    def __toHSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    def __mask(self, img):
        lower_rouille = np.array([10, 50, 50])   # orange/marron foncé
        upper_rouille = np.array([30, 255, 255])  # jaune/orange

        return cv2.inRange(img, lower_rouille, upper_rouille)
    
    def __pct_rouille(self, mask):
        nb_rouille = np.count_nonzero(mask)
        nb_non_rouille = np.sum(mask == 0)
        return nb_rouille / nb_non_rouille
    
    def __sobel_img(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  #calcule la variation de l’image (couleur, type de donnée de sortie, direction de calcul)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        sobel = cv2.magnitude(sobel_x, sobel_y) # Combinaison des 2
        del gray, sobel_x, sobel_y
        return sobel
    
    def __variance_sobel(self, img):
        sobel = self.__sobel_img(img)
        return np.var(sobel)

    def __est_delimite_par_nervures(self, img):
        # 1. Charger l'image en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Prétraitement (Flou pour réduire le bruit)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Seuillage (Thresholding) pour isoler les lésions sombres/nécrosées
        # Note : Ajustez la valeur (ici 100) selon la luminosité de vos images
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        
        # 4. Trouver les contours des lésions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0 # Aucune lésion détectée
        
        # On prend le plus grand contour (souvent la lésion principale)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 5. Obtenir la "Bounding Box" droite (Bounding Rect) et orientée (Min Area Rect)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calcul de la rectangularité (rapport entre l'aire du contour et l'aire de sa boîte)
        contour_area = cv2.contourArea(largest_contour)
        bounding_box_area = w * h
        
        if bounding_box_area == 0:
            return 0
            
        solidity = contour_area / float(bounding_box_area)
        aspect_ratio = float(h) / w if w > 0 else 0
        
        # 6. Condition de retour (Logique de décision)
        # Pour la Cercosporiose (Image 1) :
        # - La lésion est très allongée verticalement (aspect_ratio élevé)
        # - Elle remplit bien sa boîte rectangulaire car ses bords sont droits (solidity élevée)
        
        if aspect_ratio > 2.5 and solidity > 0.70:
            return 1  # La lésion est rectangulaire et suit les nervures
        else:
            return 0  # La lésion déborde (ovale, fuseau ou irrégulière)
    
    def numeriseDonnee(self, imgPath):
        img = self.__readImg(imgPath)
        imgHSV = self.__toHSV(img)
        imgMask = self.__mask(imgHSV)

        img_PCT_Rouille = self.__pct_rouille(imgMask)
        imgVariance = self.__variance_sobel(img)
        img_veinBounded = self.__est_delimite_par_nervures(img)

        data =  [img_PCT_Rouille, imgVariance, img_veinBounded]
        df = pd.DataFrame(data)
        df = df.T
        df.columns=["pct_rouille", "rugosite", "nervure"]

        return df