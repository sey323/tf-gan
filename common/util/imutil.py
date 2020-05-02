import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

"""
GANによって生成された画像を表示など．
前処理以外のGANのプログラムに影響する画像処理プログラム．
データセットの作成などの前処理系は./utilにある．
"""


def calcImageSize(height, width, stride, num):
    """
    畳み込み層のストライドの大きさと画像の畳み込みの回数から，
    再構成する画像の縦横を計算するプログラム
    """
    import math

    for i in range(num):
        height = int(math.ceil(float(height) / float(stride)))
        width = int(math.ceil(float(width) / float(stride)))
    return height, width


def color2bit(input, threshold=0):
    """
    画像を2値画像に変換するプログラム
    """
    bit = (input > threshold) * 1
    return bit


def tileImage(imgs, size=0, channel=3):
    """
    正方形に並べて表示．
    @size :size列で表示
    """
    if size == 0:
        d = int(math.sqrt(imgs.shape[0] - 1)) + 1
        size = int(math.sqrt(imgs.shape[0] - 1)) + 1
    else:
        size = size
        d = int(imgs.shape[0] / size) + 1

    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.ones((h * d, w * size, channel), dtype=np.float32) * 256
    for idx, img in enumerate(imgs):
        idx_y = int(idx / size)
        idx_x = idx - idx_y * size
        r[idx_y * h : (idx_y + 1) * h, idx_x * w : (idx_x + 1) * w, :] = img
    return r


def tileImageH(imgs, channel=3):
    """
    横に10枚づつ並べる．
    """
    col = int(len(imgs) / 10) + 1
    sep = 10 if len(imgs) > 10 else len(imgs)
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h * sep, w * col, channel), dtype=np.float32)
    for idx, img in enumerate(imgs):
        idx_x = int(idx / sep)  # 表示する列
        idx_y = idx % sep
        r[idx_y * h : (idx_y + 1) * h, idx_x * w : (idx_x + 1) * w, :] = img
    return r


def nopairImage(source, fake, channel=3):
    """
    入力画像，生成された画像を並べて表示
    """
    col = int(len(source) / 10) + 1
    sep = 10 if len(source) > 10 else len(source)
    h = source[0].shape[0] + 1
    w = source[0].shape[1] + 1
    r = np.zeros((h * sep, w * 2 * col, channel), dtype=np.float32)
    for idx, _ in enumerate(source):
        now_col = int(idx / sep) * 2  # 表示する列
        idx_y = idx % sep
        r[
            idx_y * h : h * (idx_y + 1) - 1,
            0 + (w * now_col) : w - 1 + (w * now_col),
            :,
        ] = source[idx]
        r[
            idx_y * h : h * (idx_y + 1) - 1,
            w + (w * now_col) : 2 * w - 1 + (w * now_col),
            :,
        ] = fake[idx]
    return r


def pairImage(source, fake, target, channel=3):
    """
    入力画像，出力画像，正解画像を並べて表示
    """
    col = int(len(source) / 10) + 1
    sep = 10 if len(source) > 10 else len(source)
    h = source[0].shape[0] + 1
    w = source[0].shape[1] + 1
    r = np.zeros((h * sep, w * 3 * col, channel), dtype=np.float32)
    for idx, _ in enumerate(source):
        now_col = int(idx / sep) * 3  # 表示する列
        idx_y = idx % sep
        r[
            idx_y * h : h * (idx_y + 1) - 1,
            0 + (w * now_col) : w - 1 + (w * now_col),
            :,
        ] = source[idx]
        r[
            idx_y * h : h * (idx_y + 1) - 1,
            w + (w * now_col) : 2 * w - 1 + (w * now_col),
            :,
        ] = fake[idx]
        r[
            idx_y * h : h * (idx_y + 1) - 1,
            2 * w + (w * now_col) : 3 * w - 1 + (w * now_col),
            :,
        ] = target[idx]
    return r


def saveHeat(img, grad_cam, size=0, channel=3):
    """
    入力画像とヒートマップのペアを保存
    """
    col = int(len(img) / 10) + 1
    sep = 10 if len(img) > 10 else len(img)
    h = img[0].shape[0]
    w = img[0].shape[1]
    r = np.zeros((h * sep, w * 2 * col, channel), dtype=np.float32)
    for idx, _ in enumerate(img):
        now_col = int(idx / sep) * 2  # 表示する列
        idx_y = idx % sep
        r[idx_y * h : h * (idx_y + 1), 0 + (w * now_col) : w + (w * now_col), :] = img[
            idx
        ]

        # Heatmapの作成
        cam = mkHeat(img[idx], grad_cam[idx])
        r[
            idx_y * h : h * (idx_y + 1), w + (w * now_col) : 2 * w + (w * now_col), :
        ] = cam

    return r


def mkHeat(img, grad_cam):
    h = img.shape[0]
    w = img.shape[1]

    image = np.uint8(img[:, :, ::-1] * 255.0)  # RGB -> BGR
    cam = cv2.resize(grad_cam, (h, w))  # enlarge heatmap
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)  # normalize
    cam = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )  # balck-and-white to color
    cam = np.float32(cam) + np.float32(image)  # everlay heatmap onto the image
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    return cam


def mkhist(x_axis, y_axis, save_path, title="hist", range=None):
    """
    特徴次元を折れ線グラフとして出力．
    paramnames
    ----
        x_axis : list
            横軸の値
        y_axis : list
            縦軸の値
        range  : [int, int]
            範囲
    """
    # ヒストグラムの作成と保存
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    if range is not None:
        plt.ylim(range[0], range[1])

    ax.plot(x_axis, y_axis, marker="o", markersize=1)
    fig.savefig(save_path)
    print("[COMPLETE]\tPlot Saving")
