import torch
from ultralytics import YOLO
import cv2
import numpy as np
from math import sqrt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from scipy.stats import spearmanr


class Pipeline:
    def __init__(self, weights='best.pt'):
        try:
            self.model = YOLO(weights)
            #self.chromossome = np.array()
            self.classes = {'beam':0, 'notehead':1, 'staff':2}
            self.classes_ensemble = {'beam':2, 'notehead':0, 'staff':1}
        except:
            print('Pesos inválidos')
        


    
    def predict_yolo(self, image, class_):
        results = self.model(image)

        for result in results:
            boxes = result.boxes
            masks = result.masks

        masks_filtered = []

        for box,mask in zip(boxes,masks):
            if box.cls == self.classes[class_]:
                masks_filtered.append(mask)

        resized_segmentation_masks = []

        for mask in masks_filtered:
            resized_mask = torch.nn.functional.interpolate(mask.data.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False)
            resized_segmentation_masks.append(resized_mask.data.squeeze(0))
        try:
            mask_total = (resized_segmentation_masks[0].data > 0)
            for tensor in resized_segmentation_masks:
                mask = (tensor[0].data > 0)
                mask_total = mask_total | mask
            
            mask_final = mask_total.squeeze(0)
            mask_final = mask_final.numpy()

            binary_mask = np.where(mask_final > 0.5, 1, 0)
            white_background = np.ones_like(image) * 255

            new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]
            new_image = new_image.astype(np.uint8)

            cv2.imwrite('./aoba.jpg', new_image)

            return new_image
        except:
            return np.ones((640, 640, 3), dtype=np.uint8) * 255

    def set_sam_predictor(self):

        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        #sam.to()
        self.predictor = SamPredictor(sam)


    def predict_ensemble(self, image, class_):
        
        
        results = self.model.predict(source=image, conf=0.25)

        #classes=[]
        bboxes = []
        for result in results:
            
            boxes = result.boxes
            #classes.append(result.probs)
        for box in boxes:
            if box.cls == self.classes_ensemble[class_]:
                bboxes.append(box.xyxy.tolist()[0])

        self.predictor.set_image(image)
        try:
            input_boxes = []
            #input_box = np.array(bbox)
            masks_general = []
            for bbox in bboxes:
                masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([bbox]),
                multimask_output=False,
                )
                masks_general.append(masks)
                input_boxes.append(np.array(bbox))

            mask_total = masks_general[0][0]
            for masks in masks_general:
                mask_total = mask_total | masks[0]

            binary_mask = np.where(mask_total > 0.5, 1, 0)
            white_background = np.ones_like(image) * 255

            # Apply the binary mask
            new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]
            return new_image
        except:
            return np.ones((640, 640, 3), dtype=np.uint8) * 255
    """
    def predict_CGP(self, input):
        chromossome = self.chromossome
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
                available_inputs[i+1] = self._dilate_op(available_inputs[chromossome[i][0]],all_kernels[chromossome[i][1]])
            else:
                available_inputs[i+1] = self._erode_op(available_inputs[chromossome[i][0]],all_kernels[chromossome[i][1]])
            
        return available_inputs[-1]
    """

    def image_mul(self, image1, image2):
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
        return (self.image_mul(image1, image2))/((sqrt(self.image_mul(image1, image1)))*(sqrt(self.image_mul(image2, image2))))

    # It is not commutable
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
    
    # 0 a 1

    def _fit_pearson(self, image1, image2):
        return np.corrcoef((image1.ravel(),image2.ravel()))[0,1]
    
    # -1 a 1

    def _fit_spearmanRho(self, image1, image2):
        return spearmanr(image1.ravel(),image2.ravel())[0]
    
    def metrics(self, predict, desirable):
        return {'fa':self._fit_a(predict[:,:,0], desirable[:,:,0]),
                 'fb': self._fit_b(predict[:,:,0], desirable[:,:,0]),
                 'fpearson': self._fit_pearson(predict[:,:,0], desirable[:,:,0]),
                 'fspearman': self._fit_spearmanRho(predict[:,:,0], desirable[:,:,0])}
    
    def load_chromossome(self, name):
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
        self.chromossome = np.array(final_array)

    def _binarize_image(image):
        _, image = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image // 255
    
    def _and_op(image1, image2):
        return np.logical_and(image1, image2).astype(np.uint8)

    def _or_op(image1, image2):
        return np.logical_or(image1, image2).astype(np.uint8)

    def _xor_op(image1, image2):
        return np.logical_xor(image1, image2).astype(np.uint8)

    def _not_op(image):
        return np.logical_not(image).astype(np.uint8)

    def _erode_op(image, kernel):
        return cv2.erode(image, kernel, iterations=1)

    def _dilate_op(image, kernel):
        return cv2.dilate(image, kernel, iterations=1)

    def _custom_len(arr):
        return sum(1 for x in arr if x is not None)
    




    
