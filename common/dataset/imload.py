import sys, os
import random

import numpy as np
import cv2



def _load_image(file_name, img_size=None, gray=True, is_norm=True) -> np.ndarray:
    """
    ファイル名から画像を読み込み

    Args
        file_name (str):
            読み込む画像のファイル名
        img_size ([int, int]):
            画像のファイルサイズ
        gray (boolean):
            グレイスケールに変換するかどうか
        is_norm (boolean):
            画像を正規化するかどうか

    Returns
        np.ndarray:
            np配列の画像
    """
    image = cv2.imread(file_name)

    if gray:  # グレイスケールに変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if img_size is not None:  # 画像のリサイズ
        image = cv2.resize(image, (img_size[0], img_size[1]))
        channel = 1 if gray else 3
        image = image.reshape(img_size[0], img_size[1], channel)

    if is_norm:  # 画像を0-1の範囲に正規化
        image = image.astype(np.float32) / 255.0

    assert type(image) == np.ndarray
    return image


def make(folder_name, img_size, gray=False, separate=True) -> (np.ndarray, np.ndarray):
    """
    folder_name内の画像の配列とファイル名を返す

    Args
        folder_name (str):
            読み込む画像のフォルダ名
        img_size ([int, int]):
            画像のファイルサイズ
        gray (boolean):
            グレイスケールに変換するかどうか
        separate (boolean):
            画像を分割するかどうか

    Returns
        np.ndarray:
            np配列の画像のリスト
        np.ndarray:
            画像のファイル名のリスト
    """
    channel = 1 if gray else 3
    images, file_names = np.empty((0, img_size[0], img_size[1], channel),), np.array([])

    # フォルダ内のディレクトリの読み込み
    classes = os.listdir(folder_name)

    for i, file in enumerate(classes):
        # 1枚の画像に対する処理
        if "png" not in file and "jpg" not in file:  # jpg以外のファイルは無視
            continue

        # 画像読み込み
        img = _load_image(folder_name + "/" + file, img_size=img_size, gray=gray)

        images = np.append(images, [img], axis=0)
        file_names = np.append(file_names, [file], axis=0)

    assert len(images) == len(file_names)
    return (images, file_names)
