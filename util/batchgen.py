import numpy as np
import math


class batchgen(object):
    """
    バッチを生成するクラス．
    """

    def __init__(self, image, label, size=None, channel=3, label_num=None):
        if size is None:
            width = int(math.sqrt(len(image[0]) / channel))
            height = int(math.sqrt(len(image[0]) / channel))
            size = [width, height]

        self.size = size
        self.channel = channel
        self.image = np.array(self._image2tensor(image))
        self.img_num = len(self.image)

        if isinstance(label, int) and label_num is not None:
            self.label = self._label_gen(label, label_num)
        else:
            self.label = label

        # 乱数のindex
        self.idx = None

        # バッチのスタート位置
        self.start_idx = 0
        self.end_idx = 0
        self.epoch = 0

        self.label = np.array(self.label)

    def _image2tensor(self, img):
        """
        画像をtensorflowで学習可能な形式に変換．
        """
        tensor = np.reshape(img, [len(img), self.size[0], self.size[1], self.channel])
        return tensor

    def _label_gen(self, label_idx, label_num):
        """
        onehot形式のラベルを生成する
        """
        label_onehot = np.full(self.img_num, label_idx)
        labels = np.identity(label_num)[label_onehot]
        return labels

    def getBatch(self, batch_num, idx=None):
        """
        画像とラベルをバッチサイズ分取得する．
        ---
        Parameters
            tensor  : numpy.ndarray
            label   : numpy.ndarray
        """
        self.end_idx = self.start_idx + batch_num
        # 終了位置か判定
        if self.end_idx > len(self.image):
            self.start_idx = 0
            self.end_idx = batch_num
            self.epoch += 1
            if idx is None:
                self.shuffle()
            else:
                self.shuffle(idx)

        # normalized to -0.5 ~ +0.5
        tensor, label = (
            self.image[self.start_idx : self.end_idx],
            self.label[self.start_idx : self.end_idx],
        )
        tensor = (tensor - 0.5) / 1.0

        self.start_idx += batch_num
        return tensor, label

    def shuffle(self, idx=None):
        """
        ラベルと画像を同じ乱数で並び替える．
        """
        # 乱数シードが設定されていない時
        if idx is None:
            idx = np.random.permutation(len(self.image))

        self.image = self.image[idx]
        self.label = self.label[idx]

        self.idx = idx

    def getAll(self):
        """
        全てのラベルと画像を取得
        """
        return self.image, self.label

    def getIndex(self):
        """
        シャッフルした乱数のindexを返す．
        """
        return self.idx

    def getNum(self):
        """
        バッチ内の画像のサイズを返す
        """
        return len(self.image)

    def getEpoch(self):
        """
        現在のEpochを返す．
        """
        return self.epoch
