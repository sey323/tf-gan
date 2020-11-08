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


def _create_label(label_len, label_idx, label_num) -> np.ndarray:
    """
    one_hot形式のラベルを作成する
    Args
        label_len (int):
            ラベルの種類数。
            このサイズの配列が宣言される。
        label_idx (int):
            ラベルの番号。
        label_num (int):
            このlabel_idxのラベルを幾つ作成するか
    Returns
        list:
            one_hot形式のラベル
    """
    tmp_labels = np.empty((0, label_len), int)

    # one_hot_vectorを作りラベルとして追加
    label = np.zeros(label_len)
    label[label_idx] = 1
    for i in range(label_num):
        tmp_labels = np.append(tmp_labels, [label], axis=0)

    assert type(tmp_labels) == np.ndarray

    return tmp_labels


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
    images, file_names = (
        np.empty(
            (0, img_size[0], img_size[1], channel),
        ),
        np.array([]),
    )

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


def make_labels(
    folder_name,
    img_size=[64, 64],
    gray=False,
    separate=False,
) -> (np.ndarray, np.ndarray):
    """
    folder_name内にあるフォルダの画像を読み込んで画像とラベルを返す。

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
        list:
            np配列の画像のリスト
        list:
            np配列のラベルのリスト
    """
    channel = 1 if gray else 3
    classes = os.listdir(folder_name)

    train_images, train_labels = (
        np.empty(
            (0, img_size[0], img_size[1], channel),
        ),
        np.empty((0, len(classes)), int),
    )

    # フォルダのディレクトリ=クラスとして扱う
    for i, d in enumerate(classes):  # 1つのディレクトリに対する処理
        # 画像とラベルを設定
        tmp_images, _ = make(
            folder_name + "/" + d, img_size, gray=gray, separate=separate
        )
        tmp_labels = _create_label(len(classes), i, len(tmp_images))

        train_images = np.vstack([train_images, tmp_images])
        train_labels = np.vstack([train_labels, tmp_labels])

    return train_images, train_labels
