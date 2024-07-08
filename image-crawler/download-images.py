import requests
import os
import sys
from alive_progress import alive_bar
from bs4 import BeautifulSoup as bs



def dataset_scrapping(NUM_FILE:int=100):
    counter_images = 1

    if os.path.exists('./train') == False:
        os.mkdir('./train')
    if os.path.exists('./valid') == False:
        os.mkdir('./valid')
    if os.path.exists('./test') == False:
        os.mkdir('./test')
    
    if NUM_FILE % 40 != 0:
        range_file = NUM_FILE//40 + 1
    else:
        range_file = NUM_FILE//40

    with alive_bar(NUM_FILE, force_tty=True) as bar:
        folder = 'train'
        for page in range(0,range_file):
            if page == 0:
                r = requests.get(f'https://www.superpartituras.com.br/instrumento/teclado')
            else:
                r = requests.get(f'https://www.superpartituras.com.br/instrumento/teclado/{page}')

            soup = bs(r.content,'html.parser')

            images = soup.select('div img')

            images_url = []

            for image in images:
                if 'data-src' in str(image):
                    images_url.append(image)
            
            for url in images_url:
                img_data = requests.get(url['data-src']).content
                with open(f'{folder}/{counter_images}.jpg','wb') as handler:
                    handler.write(img_data)
                    counter_images+=1
                    if counter_images > (NUM_FILE*3)//5 and counter_images <= (NUM_FILE*4)//5:
                        folder = 'valid'
                    elif counter_images > (NUM_FILE*4)//5:
                        folder = 'test'
                    bar()
                    if (counter_images > NUM_FILE):
                            return

def main():
    if (sys.argv[1] != None):
        NUM_FILE = int(sys.argv[1])
    else:
        NUM_FILE = 100
    dataset_scrapping(NUM_FILE)



if __name__=="__main__":
    main()