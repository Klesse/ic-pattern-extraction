import numpy as np
import cv2
from math import sqrt

class CGP:
    def __init__(self, chromossome_file):
        self.chromossome = self._load_chromossome(chromossome_file)
        self.all_kernels = self._set_kernels()

    def _load_chromossome(self, name):
        c = np.fromfile(f'{name}.dat', dtype=int)
        final_array = []
        aux = []
        for i in range(len(c)):
            if (i+1) % 3 == 0 and i != 0:
                aux.append(c[i])
                final_array.append(np.array(aux))
                aux = []
            else:
                aux.append(c[i])
        return np.array(final_array)
    
    def func_apply(self, input, chromossome):
        chrom_len = len(chromossome)
        available_inputs = np.empty(shape=(chrom_len+1,),dtype=object)
        available_inputs[0] = input
        
        for i in range(0, chrom_len):
            
            if (chromossome[i][2] == 11):
                available_inputs[i+1] = self._and_op(available_inputs[chromossome[i][0]],available_inputs[chromossome[i][1]]) 
            elif (chromossome[i][2] == 12):
                available_inputs[i+1] = self._or_op(available_inputs[chromossome[i][0]],available_inputs[chromossome[i][1]]) 
            elif (chromossome[i][2] == 13):
                available_inputs[i+1] = self._xor_op(available_inputs[chromossome[i][0]],available_inputs[chromossome[i][1]])  
            elif (chromossome[i][2] == 14):
                available_inputs[i+1] = self._not_op(available_inputs[chromossome[i][0]]) 
            elif (chromossome[i][2] == 15):
                available_inputs[i+1] = self._dilate_op(available_inputs[chromossome[i][0]], self.all_kernels[chromossome[i][1]])
            else:
                available_inputs[i+1] = self._erode_op(available_inputs[chromossome[i][0]], self.all_kernels[chromossome[i][1]])
            
        return available_inputs[-1]
    
    def score(self, X_test, func='pearson'):
        scores = []
        if func == 'fit_a':
            for i in range(X_test.shape[0]):
                scores.append(self._fit_a(self.func_apply(X_test['input'][i], self.chromossome), X_test['output'][i]))
        elif func == 'fit_b':
            for i in range(X_test.shape[0]):
                scores.append(self._fit_b(X_test['output'][i], self.func_apply(X_test['input'][i], self.chromossome)))
        elif func == 'pearson':
            for i in range(X_test.shape[0]):
                scores.append(self._fit_pearson(self.func_apply(X_test['input'][i], self.chromossome), X_test['output'][i]))
        else:
            print('Invalid function')
        for k in range(X_test.shape[0]):
            print(f'Image {k+1} -> {scores[k]}')

        return scores

    def _binarize_image(image, type='normal'):
        if type == 'otsu':
            _, image = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, image = cv2.threshold(image.astype(np.uint8), 200, 255, cv2.THRESH_BINARY)
        return image // 255
    
    def _and_op(self, image1, image2):
        return np.logical_and(image1, image2).astype(np.uint8)

    def _or_op(self, image1, image2):
        return np.logical_or(image1, image2).astype(np.uint8)

    def _xor_op(self, image1, image2):
        return np.logical_xor(image1, image2).astype(np.uint8)

    def _not_op(self, image):
        return np.logical_not(image).astype(np.uint8)

    def _erode_op(self, image, kernel):
        return cv2.erode(image, kernel, iterations=1)

    def _dilate_op(self, image, kernel):
        return cv2.dilate(image, kernel, iterations=1)
    
    def _image_mul(self, image1, image2):
        if (len(image1) == len(image2)):
            N = len(image1)
        else:
            print('Imagens não compatíveis: Tamanhos diferentes')
            return
        image1 = image1.astype(np.uint8)
        image2 = image2.astype(np.uint8)
        _, image1 = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, image2 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image1 = image1 // 255
        image2 = image2 // 255
        sum = 0
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                sum += image1[i][j] * image2[i][j]
        return sum*(1/N)/(1/N)



    def _fit_a(self, image2, image1):
        return (self._image_mul(image1, image2))/((sqrt(self._image_mul(image1, image1)))*(sqrt(self._image_mul(image2, image2))))
    
    def _fit_b(self, image1, image2):
        TP=0
        TN=0
        FP=0
        FN=0
        if (len(image1) == len(image2)):
            N = len(image1)
        else:
            print('Imagens não compatíveis: Tamanhos diferentes')
            return
        
        image1 = image1.astype(np.uint8)
        image2 = image2.astype(np.uint8)

        # Smart threshold
        _, image1 = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, image2 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image1 = image1 // 255
        image2 = image2 // 255
        sum = 0
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                if(image1[i][j] == 1 and image2[i][j] == 1):
                    TP+=1
                elif(image1[i][j] == 1 and image2[i][j] == 0):
                    FP+=1
                elif(image1[i][j] == 0 and image2[i][j] == 1):
                    FN+=1
                elif(image1[i][j] == 0 and image2[i][j] == 0):
                    TN+=1
        if (TP + FN == 0 or TN + FP == 0):
            SV = 1
            SP = 1
        else:
            SV = (TP)/(TP + FN)
            SP = (TN)/(TN + FP)

        return 1-((sqrt((1-SP)**2 + (1-SV)**2))/sqrt(2))
    
    def _fit_pearson(self, image1, image2):
        return np.corrcoef((image1.ravel(),image2.ravel()))[0,1]


    def _set_kernels(self):
        regular_kernels_3x3 = np.array([
        np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8),
        np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8),
        np.array([[1,0,1],[0,1,0],[1,0,1]], dtype=np.uint8),
        np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=np.uint8),
        np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.uint8),
        np.array([[0,1,0],[0,1,0],[0,1,0]], dtype=np.uint8),
        np.array([[0,0,0],[1,1,1],[0,0,0]], dtype=np.uint8),
        np.array([[1,1,1],[0,0,0],[0,0,0]], dtype=np.uint8),
        np.array([[0,0,0],[0,0,0],[1,1,1]], dtype=np.uint8)
        ])

        regular_kernels_5x5 = np.array([
            np.array(np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]], dtype=np.uint8)),
            np.array(np.array([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]], dtype=np.uint8)),
            np.array(np.array([[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]], dtype=np.uint8)),
            np.array(np.array([[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0]], dtype=np.uint8)),
            np.array(np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]], dtype=np.uint8)),
            np.array(np.array([[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]], dtype=np.uint8)),
            np.array(np.array([[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.uint8)),
            np.array(np.array([[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]], dtype=np.uint8)),
            np.array(np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]], dtype=np.uint8)),
        ])

        regular_kernels_7x7 = np.array([
            np.array(np.array([[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1], [1,1,1,1,1,1,1]], dtype=np.uint8)),
            np.array(np.array([[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[1,1,1,1,1,1,1],[0,0,0,1,0,0,0],
                            [0,0,0,1,0,0,0], [0,0,0,1,0,0,0]], dtype=np.uint8)),
            np.array(np.array([[1,0,0,0,0,0,1],[0,1,0,0,0,1,0],[0,0,1,0,1,0,0],[0,0,0,1,0,0,0],[0,0,1,0,1,0,0],
                            [0,1,0,0,0,1,0], [1,0,0,0,0,0,1]], dtype=np.uint8)),
            np.array(np.array([[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],
                            [0,0,0,1,0,0,0], [0,0,0,1,0,0,0]], dtype=np.uint8)),
            np.array(np.array([[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[1,1,1,1,1,1,1],[0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]], dtype=np.uint8)),
            np.array(np.array([[0,0,0,0,0,0,1],[0,0,0,0,0,1,0],[0,0,0,0,1,0,0],[0,0,0,1,0,0,0],[0,0,1,0,0,0,0],
                            [0,1,0,0,0,0,0], [1,0,0,0,0,0,0]], dtype=np.uint8)),
            np.array(np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],
                            [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]], dtype=np.uint8)),
            np.array(np.array([[0,0,0,1,0,0,0],[0,0,1,1,1,0,0],[0,1,1,1,1,1,0],[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],
                            [0,0,1,1,1,0,0], [0,0,0,1,0,0,0]], dtype=np.uint8)),
            np.array(np.array([[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],[0,0,1,1,1,0,0],[0,0,0,1,0,0,0],[0,0,1,1,1,0,0],
                            [0,1,1,1,1,1,0], [1,1,1,1,1,1,1]], dtype=np.uint8)),
        ])

        np.random.seed(42)

        irregular_kernels_3x3 = np.empty(shape=9,dtype=object)
        irregular_kernels_5x5 = np.empty(shape=9,dtype=object)
        irregular_kernels_7x7 = np.empty(shape=9,dtype=object)

        for i in range(9):
            irregular_kernels_3x3[i] = np.random.randint(2, size=(3, 3))
            irregular_kernels_5x5[i] = np.random.randint(2, size=(5, 5))
            irregular_kernels_7x7[i] = np.random.randint(2, size=(7, 7))

        all_kernels = regular_kernels_3x3.tolist() + \
              regular_kernels_5x5.tolist() + \
              regular_kernels_7x7.tolist() + \
              irregular_kernels_3x3.tolist() + \
              irregular_kernels_5x5.tolist() + \
              irregular_kernels_7x7.tolist()
        for idx, element in enumerate(all_kernels):
            all_kernels[idx] = np.array(element, dtype=np.uint8)

        return np.array(all_kernels, dtype=object)