import sys , os
import random

import numpy as np
import cv2

'''
一般画像からtensorflowの学習用のデータセットを作成するプログラム．
画像データは全て，numpy.ndarray形式で返される．
'''
def _save_label( labels , save_path = "label.txt"):
    '''
    ラベルデータをテキストに保存
    '''
    f = open(save_path, 'w')
    f.writelines(labels)
    f.close()


def _random_clip(img , clip_size , num = 1):
    '''
    画像をランダムの位置で切り抜くプログラム
    '''
    clip_images = []
    height, width = img.shape[:2]

    # 画像をclip_sizeサイズごとにnum回切り抜く
    for y in range( num ):
        rand_y = random.randint(0,height - clip_size)
        rand_x = random.randint(0,width - clip_size)
        clip_img = img[ rand_y : rand_y + clip_size, rand_x : rand_x + clip_size]
        clip_img = clip_img.flatten().astype(np.float32)/255.0
        clip_images.append(clip_img)

    return clip_images


def _harf_separate(img,img_size):
    '''
    入力画像を画像を半分の位置で分割する
    '''
    # pointの場所で画像を横に分割する．
    _, width = img.shape[:2]
    point = int(width/2)
    source_image = img[:, :point, :]
    source_image = cv2.resize( source_image , (img_size , img_size ))

    target_image = img[:, point:, :]
    target_image = cv2.resize( target_image , (img_size , img_size ))
    return source_image , target_image


def _random_3sampling(*args,train_num,test_num = 0  ):
    '''
    データセットから画像とラベルをランダムに取得
    ---
    Parameters
        *args       : array
            対応関係を持ったままランダムに並び替える配列
        train_num   : int
            学習データとする画像の枚数
        test_num    : int
            テストデータとする画像の枚数
    '''

    zipped = list(zip(*args))
    #乱数を発生させ，リストを並び替える．
    np.random.shuffle(zipped)

    if train_num == 0:
        train_zipped = zipped[:]
    else:
        train_zipped = zipped[:train_num]

    train_zipped = list(zip(*train_zipped))
    # Numpy配列に再変換
    train_ary = []
    for ar in train_zipped:
        train_ary.append(np.asarray(ar))

    if test_num == 0: # 検証用データの指定がないとき
        return train_ary
    else:
        test_zipped = zipped[ train_num : train_num + test_num ]
        test_zipped = list(zip(*test_zipped))

        test_ary = []
        for ar in test_zipped:
            test_ary.append(np.asarray(ar))

        return train_ary,test_ary


def make( folder_name , separate=False ,gray = False , img_size = 0 ,train_num = 0 , test_num = 0, clip_num = 0 , clip_size = 0):
    '''
    画像フォルダを読み込む
    ---
    Parameters
        folder_name : String
            読み込む画像フォルダのパス
        separate    : Boolean
            画像を分割するかどうか
        gray        : Boolean
            2値化処理を行うかどうか
        img_size    : int
            画像サイズの1辺
        train_num   : int
            学習データとする画像の枚数
        test_num    : int
            テストデータとする画像の枚数
        clip_num    : int
            ランダムクリップを行う際に何枚画像を切り抜くか
        clip_size   : int
            ランダムクリップを行う際の画像サイズ
    '''

    train_images = []
    train_labels = []
    target_images = []
    labels = ""

    # フォルダ内のディレクトリの読み込み
    classes = os.listdir( folder_name )

    # フォルダのディレクトリ=クラスとして扱う
    for i, d in enumerate(classes):# 1つのディレクトリに対する処理
        files = os.listdir( folder_name + '/' + d  )
        tmp_image = []
        tmp_label = []
        for j,file in enumerate(files):# 1つのファイルに対する処理
            if not 'png' in file and not 'jpg' in file and not 'jpeg' in file:# jpg以外のファイルは無視
                continue
            # 画像読み込み
            img = cv2.imread( folder_name+ '/' + d + '/' + file )
            if img is None:
                continue

            # one_hot_vectorを作りラベルとして追加
            label = np.zeros(len(classes))
            label[i] = 1
            # 画像を分割するとき
            if separate:
                source_image,target_image =_harf_separate(img,img_size)
                if gray:#グレイスケールに変換
                    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
                    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

                source_image = source_image.flatten().astype(np.float32)/255.0
                target_image = target_image.flatten().astype(np.float32)/255.0

                train_images.append( source_image )
                target_images.append( target_image )
                train_labels.append( label )
                continue

            if gray:#グレイスケールに変換
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ランダム位置でクリップを行うとき．
            if clip_size != 0 and clip_num != 0:
                img = _random_clip( img , clip_size , clip_num)
                tmp_image.extend( img )
                for j in range(clip_num):
                    tmp_label.append(label)
            else:# ランダムクリップをしないときは，リサイズをして保存
                if img_size != 0:# リサイズをする．
                    img = cv2.resize( img , (img_size , img_size ))

            img = img.flatten().astype(np.float32)/255.0
            tmp_image.append(img)
            tmp_label.append(label)
            train_images.extend( tmp_image )
            train_labels.extend( tmp_label )

        labels += "label,{0},name,{1}".format(i , d)+"\n"
        print('[LOADING]\tLabel' + str(i) + '\tName:' + d + '\tPictures exit. Unit On '+ str(j))

    _save_label(labels,save_path=folder_name+'/../label.txt')

    if separate:
        if test_num != 0 :
            train_batch,test_batch = _random_3sampling( train_images , target_images ,train_labels, train_num=train_num , test_num =test_num )
            return train_batch[0],train_batch[1],train_batch[2],test_batch[0],test_batch[1],test_batch[2]
        else:
            train_batch = _random_3sampling( train_images , target_images ,train_labels, train_num=train_num )
            return train_batch[0],train_batch[1],train_batch[2]
    else:
        if test_num != 0 :
            train_batch,test_batch = _random_3sampling( train_images , train_labels ,train_num=train_num , test_num =test_num )
            return train_batch[0],train_batch[1],test_batch[0],test_batch[1]
        else:
            train_batch = _random_3sampling( train_images , train_labels ,train_num=train_num )
            return train_batch[0],train_batch[1]
