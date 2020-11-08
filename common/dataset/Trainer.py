import os
import time
import numpy as np


class Trainer:
    """
    モデルの学習を行う
    """

    def __init__(
        self,
        batch_num=50,
        max_epoch=10000,
    ):
        self.max_epoch = max_epoch
        self.batch_num = batch_num

    def train(self, batch_o, model, type="gan"):
        if type != "gan":
            # 画像を生成するラベルの作成．
            self.test_code = []
            for i in range(self.batch_num):
                tmp = np.zeros(model.label_num)
                tmp.put(i % model.label_num, 1)
                self.test_code.append(tmp)

        epoch = 0
        step = -1
        time_history = []
        start = time.time()
        while batch_o.getEpoch() <= self.max_epoch:
            step += 1

            # ランダムにバッチと画像を取得
            batch_images, batch_labels = batch_o.getBatch(self.batch_num)
            batch_z = np.random.uniform(
                -1.0, +1.0, [self.batch_num, model.noise_dim]
            ).astype(np.float32)
            if type == "gan":
                update_result_dict = model.update(batch_z, batch_images)
            else:
                update_result_dict = model.update(batch_z, batch_images, batch_labels)
            if step > 0 and step % 10 == 0:
                model.add_summary(update_result_dict["summary"], step)

            # Epoch毎に画像を保存．
            if epoch != batch_o.getEpoch():
                # 実行時間と損失関数の出力
                train_time = time.time() - start
                time_history.append({"step": step, "time": train_time})
                print(
                    "%6d: loss(D)=%.4e, loss(G)=%.4e; time/step=%.2f sec"
                    % (
                        step,
                        update_result_dict["d_loss"],
                        update_result_dict["g_loss"],
                        train_time,
                    )
                )
                model.dumper.add(
                    batch_o.getEpoch(),
                    step,
                    update_result_dict["d_loss"],
                    update_result_dict["g_loss"],
                    train_time,
                )

                # ノイズの作成．
                z1 = np.random.uniform(-1, +1, [self.batch_num, model.noise_dim])
                z2 = np.random.uniform(-1, +1, [model.noise_dim])
                z2 = np.expand_dims(z2, axis=0)
                z2 = np.repeat(z2, repeats=self.batch_num, axis=0)

                # 画像を作成して保存
                if type == "gan":
                    model.create(
                        z1,
                        os.path.join(
                            model.save_folder, "images", "img_%d_fake1.png" % epoch
                        ),
                    )
                    model.create(
                        z2,
                        os.path.join(
                            model.save_folder, "images", "img_%d_fake2.png" % epoch
                        ),
                    )
                else:
                    model.create(
                        z1,
                        os.path.join(
                            model.save_folder, "images", "img_%d_fake1.png" % step
                        ),
                        self.test_code,
                    )
                    model.create(
                        z2,
                        os.path.join(
                            model.save_folder, "images", "img_%d_fake2.png" % step
                        ),
                        self.test_code,
                    )
                model.save(epoch)

                # 時間の計測の再開
                start = time.time()
                epoch = batch_o.getEpoch()

        model.dumper.save()
