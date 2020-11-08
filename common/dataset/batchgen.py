import numpy as np


class batchgen(object):
    """バッチを生成するクラス
    与えられた画像とラベルの組み合わせからDeepLearningの学習バッチを作成する。

    """

    def __init__(
        self, image, label,
    ):
        self.image = np.array(image)
        self.label = np.array(label)

        self.idx = None  # 乱数のindex

        self._start_idx = 0  # バッチの開始位置
        self._end_idx = 0  # バッチの終了位置
        self.epoch = 0  # 現在のエポック数

        # エラーチェック
        assert len(self.image) == len(self.label)

    def getBatch(self, batch_num, idx=None, pair_label=None):
        """
        画像とラベルをバッチサイズ分取得する．
        type@ tensor : numpy.ndarray
        type@ label : numpy.ndarray
        """
        self._end_idx = self._start_idx + batch_num
        # 終了位置か判定
        if self._end_idx > len(self.image):
            self._start_idx = 0
            self._end_idx = batch_num
            self.epoch += 1
            if idx is None:
                self.shuffle()
            else:
                self.shuffle(idx)

            if pair_label is not None:
                self.shuffleRand(pair_label)

        # normalized to -0.5 ~ +0.5
        tensor, label = (
            self.image[self._start_idx : self._end_idx],
            self.label[self._start_idx : self._end_idx],
        )

        self._start_idx += batch_num
        return tensor, label

    def shuffle(self, idx=None):
        """
        ラベルと画像を同じ乱数で並び替える．
        """
        if idx is None:  # 乱数シードが設定されていない時
            idx = np.random.permutation(len(self.image))

        self.image = self.image[idx]
        self.label = self.label[idx]

        self.idx = idx

    def getAll(self) -> (np.ndarray, np.ndarray):
        """
        全てのラベルと画像を取得
        """
        return self.image, self.label

    def getIndex(self) -> np.ndarray:
        """
        シャッフルした乱数のindexを返す．
        """
        return self.idx

    def getNum(self) -> int:
        """
        バッチ内の画像の枚数を返す
        """
        return len(self.image)

    def getEpoch(self) -> int:
        """
        現在のEpochを返す．
        """
        return self.epoch
