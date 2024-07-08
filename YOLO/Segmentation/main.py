import pipeline as pl
import os
import numpy as np
import pandas as pd
import cv2
from time import perf_counter

def count_class():
    pass

def input_images(path):

    images = []

    files_names = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    files_names.sort()
    for file_name in files_names:

        image = cv2.cvtColor(cv2.imread(f"{path}/{file_name}"), cv2.COLOR_BGR2RGB)
        images.append(image)
    
    return images
        
def test_images(class_):

    path = './test/'+class_  

    images = input_images(path)
    
    
    return images

def binarize_image(image):
    _, image = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image // 255

def custom_set_CGP(classes, type='train'):
    if type == 'train':
        path_input = './train/input/'
        path_output = './train/output/'
    else:
        path_input = './test/input/'
        path_output = './test/output/'
    dataframe = pd.DataFrame(columns=['input', 'output'])
    for class_ in classes:
        for f in sorted(os.listdir(path_input)):
            dataframe = dataframe._append({'input': binarize_image(cv2.imread(os.path.join(path_input, f))[:,:,0]), 
                                           'output': binarize_image(cv2.imread(os.path.join(path_output+class_, f))[:,:,0]),
                                           'class': class_},
                                           ignore_index=True)
            
    return dataframe
    

def main():
    test_images_ = [1,2,3,4,5,6,7,8,9,10]
    metrics_yolo = pd.DataFrame(index=test_images_, columns=['fa_beam', 'fb_beam', 
                                                             'fpearson_beam', 'fspearman_beam',
                                                             'fa_notehead', 'fb_notehead',
                                                             'fpearson_notehead', 'fspearman_notehead'
                                                             ])
    statistics_yolo = {'mean_fa_beam': 0,
                       'mean_fb_beam':0,
                       'mean_fpearson_beam'
                       'mean_fspearman_beam':0,
                       'mean_fa_notehead':0,
                       'mean_fb_notehead':0,
                       'mean_fpearson_notehead':0,
                       'mean_fspearman_notehead':0,
                       'std_fa_beam': 0,
                       'std_fb_beam':0,
                       'std_fpearson_beam':0,
                       'std_fspearman_beam':0,
                       'std_fa_notehead':0,
                       'std_fb_notehead':0,
                       'std_pearson_notehead':0,
                       'std_spearman_notehead':0
                       }
    temp_exec_yolo = 0


    # YOLOV8
    model = pl.Pipeline('best_predict.pt')
    images = input_images('./test')

    i = 0
    t1_start = perf_counter()
    for class_ in ['beam', 'notehead']:
        tests = test_images(class_)
        for input, test_ in zip(images, tests):
            predict = model.predict_yolo(input, class_)
            #_, predict = cv2.threshold(predict, 127, 255, cv2.THRESH_BINARY) # Threshold to be binary
            metrics_dict = model.metrics(predict, test_)
            metrics_yolo.iloc[i]['fa_'+class_] = metrics_dict['fa']
            metrics_yolo.iloc[i]['fb_'+class_] = metrics_dict['fb']
            metrics_yolo.iloc[i]['fpearson_'+class_] = metrics_dict['fpearson']
            metrics_yolo.iloc[i]['fspearman_'+class_] = metrics_dict['fspearman']
            i+=1
            if i > 9:
                i = 0
    temp_exec_yolo = perf_counter() - t1_start
    metrics_yolo.to_csv('yolo.csv')

    statistics_yolo['mean_fa_beam'] = metrics_yolo['fa_beam'].mean()
    statistics_yolo['mean_fb_beam'] = metrics_yolo['fb_beam'].mean()
    statistics_yolo['mean_fpearson_beam'] = metrics_yolo['fpearson_beam'].mean()
    statistics_yolo['mean_fspearman_beam'] = metrics_yolo['fspearman_beam'].mean()
    statistics_yolo['mean_fa_notehead'] = metrics_yolo['fa_notehead'].mean()
    statistics_yolo['mean_fb_notehead'] = metrics_yolo['fb_notehead'].mean()
    statistics_yolo['mean_fpearson_notehead'] = metrics_yolo['fpearson_notehead'].mean()
    statistics_yolo['mean_fspearman_notehead'] = metrics_yolo['fspearman_notehead'].mean()

    statistics_yolo['std_fa_beam'] = metrics_yolo['fa_beam'].std()
    statistics_yolo['std_fb_beam'] = metrics_yolo['fb_beam'].std()
    statistics_yolo['std_fpearson_beam'] = metrics_yolo['fpearson_beam'].std()
    statistics_yolo['std_fspearman_beam'] = metrics_yolo['fspearman_beam'].std()
    statistics_yolo['std_fa_notehead'] = metrics_yolo['fa_notehead'].std()
    statistics_yolo['std_fb_notehead'] = metrics_yolo['fb_notehead'].std()
    statistics_yolo['std_fpearson_notehead'] = metrics_yolo['fpearson_notehead'].std()
    statistics_yolo['std_fspearman_notehead'] = metrics_yolo['fspearman_notehead'].std()
    print('\nYOLOv8')
    print(metrics_yolo)
    print(statistics_yolo)
    print(f"YOLOv8 -> {temp_exec_yolo} [s]")
    print()


    # Ensemble

    model = pl.Pipeline('best_ensemble_final.pt')
    model.set_sam_predictor()
    metrics_ensemble = pd.DataFrame(index=test_images_, columns=['fa_beam', 'fb_beam', 
                                                             'fpearson_beam', 'fspearman_beam',
                                                             'fa_notehead', 'fb_notehead',
                                                             'fpearson_notehead', 'fspearman_notehead'
                                                             ])
    
    statistics_ensemble = {'mean_fa_beam': 0,
                       'mean_fb_beam':0,
                       'mean_fpearson_beam':0,
                       'mean_fspearman_beam':0,
                       'mean_fa_notehead':0,
                       'mean_fb_notehead':0,
                       'mean_fpearson_notehead':0,
                       'mean_fspearman_notehead':0,
                       'std_fa_beam': 0,
                       'std_fb_beam':0,
                       'std_fpearson_beam':0,
                       'std_fspearman_beam':0,
                       'std_fa_notehead':0,
                       'std_fb_notehead':0,
                       'std_fpearson_notehead':0,
                       'std_fspearman_notehead':0
                       }
    

    temp_exec_ensemble = 0


    images = input_images('./test')

    i=0

    t1_start = perf_counter()
    for class_ in ['beam', 'notehead']:
        tests = test_images(class_)
        for input, test_ in zip(images, tests):
            predict = model.predict_ensemble(input, class_)
            metrics_dict = model.metrics(predict, test_)
            metrics_ensemble.iloc[i]['fa_'+class_] = metrics_dict['fa']
            metrics_ensemble.iloc[i]['fb_'+class_] = metrics_dict['fb']
            metrics_ensemble.iloc[i]['fpearson_'+class_] = metrics_dict['fpearson']
            metrics_ensemble.iloc[i]['fspearman_'+class_] = metrics_dict['fspearman']

            i+=1
            if i > 9:
                i = 0
    temp_exec_ensemble = perf_counter() - t1_start

    metrics_ensemble.to_csv('ensemble.csv')

    statistics_ensemble['mean_fa_beam'] = metrics_ensemble['fa_beam'].mean()
    statistics_ensemble['mean_fb_beam'] = metrics_ensemble['fb_beam'].mean()
    statistics_ensemble['mean_fpearson_beam'] = metrics_ensemble['fpearson_beam'].mean()
    statistics_ensemble['mean_fspearman_beam'] = metrics_ensemble['fspearman_beam'].mean()
    statistics_ensemble['mean_fa_notehead'] = metrics_ensemble['fa_notehead'].mean()
    statistics_ensemble['mean_fb_notehead'] = metrics_ensemble['fb_notehead'].mean()
    statistics_ensemble['mean_fpearson_notehead'] = metrics_ensemble['fpearson_notehead'].mean()
    statistics_ensemble['mean_fspearman_notehead'] = metrics_ensemble['fspearman_notehead'].mean()

    statistics_ensemble['std_fa_beam'] = metrics_ensemble['fa_beam'].std()
    statistics_ensemble['std_fb_beam'] = metrics_ensemble['fb_beam'].std()
    statistics_ensemble['std_fpearson_beam'] = metrics_ensemble['fpearson_beam'].std()
    statistics_ensemble['std_fspearman_beam'] = metrics_ensemble['fspearman_beam'].std()
    statistics_ensemble['std_fa_notehead'] = metrics_ensemble['fa_notehead'].std()
    statistics_ensemble['std_fb_notehead'] = metrics_ensemble['fb_notehead'].std()
    statistics_ensemble['std_fpearson_notehead'] = metrics_ensemble['fpearson_notehead'].std()
    statistics_ensemble['std_fspearman_notehead'] = metrics_ensemble['fspearman_notehead'].std()
    
    print('\nENSEMBLE (YOLOv8 + SAM)')
    print(metrics_ensemble)
    print(statistics_ensemble)
    print(f"ENSEMBLE -> {temp_exec_ensemble} [s]")
    print()
    print("COLOCAR MAX A MIN")



def tests():
    model = pl.Pipeline('best_ensemble_final.pt')
    model.set_sam_predictor()
    test_images_ = [1,2,3,4,5,6,7,8,9,10]
    metrics_ensemble = pd.DataFrame(index=test_images_, columns=['fa_beam', 'fb_beam',
                                                       'fa_notehead', 'fb_notehead'])

    images = input_images('./test')

    i=0

    for class_ in ['beam', 'notehead']:
        tests = test_images(class_)
        for input, test_ in zip(images, tests):
            predict = model.predict_ensemble(input, class_)
            metrics_dict = model.metrics(predict, test_)
            metrics_ensemble.iloc[i]['fa_'+class_] = metrics_dict['fa']
            metrics_ensemble.iloc[i]['fb_'+class_] = metrics_dict['fb']
            i+=1
            if i > 9:
                i = 0
    print(metrics_ensemble)


if __name__=="__main__":
    main()
    #tests()