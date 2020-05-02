import cv2
import json
import logging
import os
import sys
import tensorflow as tf
from datetime import datetime

tf.compat.v1.disable_eager_execution()

sys.path.append("./")
import common.layer as layer
import common.loss_function as loss_function
import common.optimizer as optimizer
import common.util.imutil as imutil
from common.util import Dumper


class GAN(object):
    """
    GANの基本クラス
    """

    def __init__(
        self,
        input_size,
        channel=3,
        layers=[64, 128, 256,],
        filter_size=[5, 5],
        drop_prob=0.5,
        noise_dim=100,
        learn_rate=2e-4,
        gpu_config=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1),
        save_folder="results/GAN",
    ):
        self.input_size = input_size
        self.channel = channel
        self.layers = layers
        self.filter_size = filter_size
        self.drop_prob = drop_prob
        self.noise_dim = noise_dim
        self.learn_rate = learn_rate

        # 保存するディレクトリの作成．
        now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.save_folder = save_folder + "/" + now

        if self.save_folder and not os.path.exists(
            os.path.join(self.save_folder, "images")
        ):
            os.makedirs(os.path.join(self.save_folder, "images"))
            os.makedirs(os.path.join(self.save_folder, "eval"))
            param_file = open(self.save_folder + "/param.json", "w")
            # パラメータをJsonファイルに保存
            json.dump(self.__dict__, param_file, indent=2)
            self.dumper = Dumper(
                "epoch",
                "step",
                "D_loss",
                "G_loss",
                "train_time",
                save_path=self.save_folder,
            )

        # GPUの設定
        self.gpu_config = tf.compat.v1.ConfigProto(gpu_options=gpu_config)
        logging.basicConfig(level=logging.DEBUG)
        print("Tensorflow Versions : {}".format(tf.__version__))

        self.build_model()
        self.init_session()
        self.init_summry()

    def output_layer(self, input, channel=3):
        """
        画像を再構成
        """
        with tf.compat.v1.variable_scope("image_reconstract") as scope:
            deconv_out = layer.deconv2d(
                input,
                stride=2,
                filter_size=[self.filter_size[0], self.filter_size[1]],
                output_shape=[self.input_size[0], self.input_size[1]],
                output_dim=channel,
                batch_norm=False,
                name="Deconv_Output",
            )
            output = layer.tanh(deconv_out)
        return output

    def Generator(self, input, channel=3, reuse=False):
        """
        Generator

        Args
            input (tensor):
                生成の基となるランダムノイズ
            channel (int):
                出力するカラーチャネル
            reuse (Boolean):
                同じネットワークが呼び出された時に再定義し直すかどうか
        """
        logging.info("[NETWORK]\tDeep Convolutional Generator")
        with tf.compat.v1.variable_scope("Generator", reuse=reuse) as scope:
            if reuse:
                scope.reuse_variables()

            # 逆FC
            with tf.compat.v1.variable_scope("Fc{0}".format("")) as scope:
                dim_h, dim_w = imutil.calcImageSize(
                    self.input_size[0],
                    self.input_size[1],
                    stride=2,
                    num=len(self.layers),
                )
                # 1層目
                defc_1 = layer.defc(
                    input,
                    output_shape=[dim_h, dim_w,],
                    output_dim=self.layers[-1],
                    name="defc",
                )
            before_output = defc_1

            # Deconv層
            for i, input_shape in enumerate(reversed(self.layers)):
                # 初期情報
                layer_no = len(self.layers) - i
                output_dim = self.layers[layer_no - 1]
                output_h, output_w = imutil.calcImageSize(
                    self.input_size[0], self.input_size[1], stride=2, num=layer_no
                )
                logging.debug(
                    "[OUTPUT]\t(batch_size, output_height:{0}, output_width:{1}, output_dim:{2})".format(
                        output_h, output_w, output_dim
                    )
                )

                # deconv
                with tf.compat.v1.variable_scope(
                    "deconv_layer{0}".format(layer_no)
                ) as scope:
                    deconv = layer.deconv2d(
                        before_output,
                        stride=2,
                        filter_size=[self.filter_size[0], self.filter_size[1]],
                        output_shape=[output_h, output_w],
                        output_dim=output_dim,
                        batch_norm=True,
                        name="Deconv_{}".format(layer_no),
                    )
                    before_output = layer.ReLU(deconv)

            # Outputで画像に復元
            output = self.output_layer(before_output, channel=channel)
        return output

    def Discriminator(self, input, channel=3, reuse=False, name=""):
        """
        Discriminator

        Args
            input (tensor):
                本物か偽物か識別したい画像のTensor配列
            channel (int):
                入力するカラーチャネル
            reuse (Boolean):
                同じネットワークが呼び出された時に再定義し直すかどうか
        """
        logging.info("[NETWORK]\tDeep Convolutional Discriminator")
        with tf.compat.v1.variable_scope("Discriminator" + name, reuse=reuse) as scope:
            if reuse:
                scope.reuse_variables()

            for i, output_shape in enumerate(self.layers, 1):
                if i == 1:  # 1層目の時だけ
                    before_output = input
                # conv
                with tf.compat.v1.variable_scope("conv_layer{0}".format(i)) as scope:
                    conv = layer.conv2d(
                        input=before_output,
                        stride=2,
                        filter_size=[self.filter_size[0], self.filter_size[1]],
                        output_dim=output_shape,
                        batch_norm=True,
                        name="Conv_{}".format(i),
                    )
                    conv = layer.leakyReLU(conv)

                before_output = conv
        return before_output

    def FcDiscriminator(self, input, channel=3, reuse=False):
        logging.info("[NETWORK]\tFlatten")
        disc_output = self.Discriminator(input=input, channel=channel, reuse=reuse)
        with tf.compat.v1.variable_scope("Discriminator_Flatten", reuse=reuse) as scope:
            # FC層
            flatten_1 = layer.flatten(disc_output, "Flatten")
            output = layer.fc(flatten_1, 1, "Out", batch_norm=False)
        return output

    def build_model(self):
        """
        ネットワークの全体を作成する
        """

        """変数の定義"""
        self.z = tf.compat.v1.placeholder(tf.float32, [None, self.noise_dim], name="z")
        self.y_real = tf.compat.v1.placeholder(
            tf.float32, [None, self.input_size[0], self.input_size[1], 3], name="image"
        )

        """Generatorのネットワークの構築"""
        logging.info("[BUILDING]\tGenerator")
        self.y_fake = self.Generator(self.z, self.channel)
        self.y_sample = self.Generator(self.z, self.channel, reuse=True)

        """Discrimnatorのネットワークの構築"""
        logging.info("[BUILDING]\tDiscriminator")
        self.d_real = self.FcDiscriminator(self.y_real, channel=self.channel)
        self.d_fake = self.FcDiscriminator(
            self.y_fake, channel=self.channel, reuse=True
        )

        """損失関数の定義"""
        logging.info("[BUILDING]\tLoss Function")
        self.g_loss = loss_function.cross_entropy(
            x=self.d_fake, labels=tf.ones_like(self.d_fake), name="g_loss_fake",
        )

        self.d_loss_real = loss_function.cross_entropy(
            x=self.d_real, labels=tf.ones_like(self.d_real), name="d_loss_real",
        )
        self.d_loss_fake = loss_function.cross_entropy(
            x=self.d_fake, labels=tf.zeros_like(self.d_fake), name="d_loss_fake",
        )
        self.d_loss = self.d_loss_real + self.d_loss_fake

        """最適化関数の定義"""
        logging.info("[BUILDING]\tOptimizer")
        self.g_optimizer = optimizer.adam(self.g_loss, self.learn_rate, "Generator")
        self.d_optimizer = optimizer.adam(self.d_loss, self.learn_rate, "Discriminator")

    def update(self, batch_z, batch_images):
        # Update Discrimnator
        _, d_loss, _, _, summary = self.sess.run(
            [self.d_optimizer, self.d_loss, self.y_fake, self.y_real, self.summary],
            feed_dict={self.z: batch_z, self.y_real: batch_images},
        )
        # Update Generator
        _, g_loss = self.sess.run(
            [self.g_optimizer, self.g_loss], feed_dict={self.z: batch_z}
        )
        return {"g_loss": g_loss, "d_loss": d_loss, "summary": summary}

    def restore(self, file_name):
        """
        Restore Model

        Args
            file_name (String):
                読み込みたいモデルのパス
        """
        self.build_model()
        self.init_session()
        # モデルファイルが存在するかチェック
        ckpt = tf.train.get_checkpoint_state(file_name)
        if ckpt:
            print("[LOADING]\t" + file_name)
            ckpt_name = file_name + "/" + ckpt.model_checkpoint_path.split("/")[-1]
            self.saver.restore(self.sess, ckpt_name)
            print("[LOADING]\t" + ckpt_name + " Complete!")
        else:
            print(file_name + " Not found")
            exit()

    def init_session(self):
        """
        TFセッションの初期化
        """
        self.sess = tf.compat.v1.Session(config=self.gpu_config)
        initOP = tf.compat.v1.global_variables_initializer()
        self.sess.run(initOP)

    def init_summry(self):
        """
        Tensorboadで確認可能な実行ログを記録するSaverの作成
        """
        self.saver = tf.compat.v1.train.Saver()
        self.summary = tf.compat.v1.summary.merge_all()
        if self.save_folder:
            self.writer = tf.compat.v1.summary.FileWriter(
                self.save_folder, self.sess.graph
            )

    def add_summary(self, summary, step):
        """
        実行ログを保存する
        """
        self.writer.add_summary(summary, step)

    def create(self, noize, save_folder="", label=None):
        """
        学習モデルから画像の生成

        Args
            noize ([float32]):
                生成するランダムノイズ
        """
        if label:
            g_image = self.sess.run(
                self.y_sample, feed_dict={self.z: noize, self.label: label}
            )
        else:
            g_image = self.sess.run(self.y_sample, feed_dict={self.z: noize})

        if save_folder == "":
            save_folder = self.save_folder + "/create_img.jpg"
        cv2.imwrite(save_folder, imutil.tileImage(g_image * 255.0 + 128.0))
