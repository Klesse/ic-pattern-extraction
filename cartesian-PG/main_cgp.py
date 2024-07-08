import CGP
import numpy as np
import pandas as pd
import os
import cv2

np.seterr(all="ignore")

def binarize_image(image, type='normal'):
    if type == 'otsu':
        _, image = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, image = cv2.threshold(image.astype(np.uint8), 200, 255, cv2.THRESH_BINARY)
    return image // 255

def custom_set(classes, type='train'):
    if type == 'train':
        path_input = './train/input/'
        path_output = './train/output/'
    else:
        path_input = './test/input/'
        path_output = './test/output/'
    dataframe = pd.DataFrame(columns=['input', 'output'])
    for class_ in classes:
        for f in sorted(os.listdir(path_input)):
            dataframe = dataframe._append({'input': binarize_image(cv2.imread(os.path.join(path_input, f))[:,:,0], 'otsu'), 
                                           'output': binarize_image(cv2.imread(os.path.join(path_output+class_, f))[:,:,0], 'otsu'),
                                           'class': class_},
                                           ignore_index=True)
            
    return dataframe

def main():
    
    CGP_ = CGP.CGP(f'note_90_25_005_350_final')
    for class_ in ['beam']:
        X_test = custom_set([class_], type='test')
        print(f'+=+=+=+=+=+=+=+=+= {class_.upper()} =+=+=+=+=+=+=+=+=+')
        if class_ == 'beam':
            CGP_ = CGP.CGP(f'beam_90_35_005_350_final')
        for func in ['fit_b','pearson']:
            CGP_.score(X_test, func)
            print()

if __name__ == '__main__':
    main()