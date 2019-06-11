import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

'''
GANによって生成された画像を表示など．
前処理以外のGANのプログラムに影響する画像処理プログラム．
データセットの作成などの前処理系は./utilにある．
'''

def calcImageSize( height , width ,stride , num ):
    '''
    畳み込み層のストライドの大きさと画像の畳み込みの回数から，
    再構成する画像の縦横を計算するプログラム
    '''
    import math
    for i in range(num):
        height = int(math.ceil(float(height)/float(stride)))
        width = int(math.ceil(float(width)/float(stride)))
    return height , width


def tileImage( imgs , size = 0,channel= 3):
    '''
    正方形に並べて表示．
    @size :size列で表示
    '''
    if size == 0:
        d = int(math.sqrt(imgs.shape[0]-1))+1
        size = int(math.sqrt(imgs.shape[0]-1))+1
    else:
        size = size
        d = int(imgs.shape[0]/size)+1

    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.ones((h*d,w*size,channel),dtype=np.float32)*256
    for idx,img in enumerate(imgs):
        idx_y = int(idx/size)
        idx_x = idx-idx_y*size
        r[ idx_y*h : ( idx_y + 1 ) * h , idx_x * w:( idx_x + 1 ) * w ,:] = img
    return r
