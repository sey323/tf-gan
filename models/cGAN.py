import sys
import logging
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

sys.path.append("./")
from models.GAN import GAN
import common.loss_function as loss_function
import common.optimizer as optimizer


class cGAN(GAN):
    """
    ConditionalGAN
    """

    def __init__(
        self,
        input_size,
        channel,
        label_num,
        layers=[64, 128, 256, 512,],
        filter_size=[5, 5],
        drop_prob=0.5,
        noise_dim=100,
        learn_rate=2e-4,
        gpu_config=tf.compat.v1.GPUOptions(allow_growth=True),
        save_folder="results/ConditionalGan",
    ):
        self.label_num = label_num
        super().__init__(
            input_size=input_size,
            channel=channel,
            layers=layers,
            filter_size=filter_size,
            drop_prob=drop_prob,
            noise_dim=noise_dim,
            learn_rate=learn_rate,
            gpu_config=gpu_config,
            save_folder=save_folder,
        )
        self.number = 2
        self.name = "Conditional GANs"


    def ConditionalGenerator(
        self, input, label, layers, output_shape, channel=3, filter_size=[3, 3], reuse=False
    ):
        """
        Generator
        """
        logging.info("\t[NETWROK]\tNoise Conditional Generator")
        # ノイズにラベルを加算
        concat_input = tf.concat([input, label], axis=1, name="concat_z")

        output = self.Generator(
            concat_input,
            channel=channel,
            reuse=reuse,
        )

        return output


    def ConditionalDiscriminator(
        self, input, label, layers, channel=3, filter_size=[3, 3], reuse=False, name=""
    ):
        """
        Discriminator
        """
        logging.info("\t[NETWROK]\tNoise Conditional Discriminator")

        label_num = label.get_shape()[-1]

        with tf.compat.v1.variable_scope("Noise_Concat", reuse=reuse) as scope:
            if reuse:
                scope.reuse_variables()
            label = tf.reshape(label, [-1, 1, 1, label_num])
            output_shape = tf.stack(
                [tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], label_num,],
            )
            k = tf.ones(output_shape) * label
            concat_input = tf.concat([input, k], axis=3)

        output = self.Discriminator(
            input=concat_input,
            channel=channel,
            reuse=reuse,
            name=name,
        )

        return output

    def build_model(self):
        """
        モデルの生成
        """
        # 変数の定義
        self.z = tf.compat.v1.placeholder(tf.float32, [None, self.noise_dim], name="z")
        self.label = tf.compat.v1.placeholder(
            tf.float32, [None, self.label_num], name="label"
        )
        self.y_real = tf.compat.v1.placeholder(
            tf.float32,
            [None, self.input_size[0], self.input_size[1], self.channel],
            name="image",
        )

        # Generatorのネットワークの構築
        logging.info("[BUILDING]\tGenerator")
        self.y_fake = self.ConditionalGenerator(
            self.z,
            self.label,
            layers=self.layers,
            output_shape=self.input_size,
            channel=self.channel,
            filter_size=self.filter_size,
        )
        self.y_sample = self.ConditionalGenerator(
            self.z,
            self.label,
            layers=self.layers,
            output_shape=self.input_size,
            channel=self.channel,
            filter_size=self.filter_size,
            reuse=True,
        )

        # Discrimnatorのネットワークの構築
        logging.info("[BUILDING]\tGenerator")
        self.d_real = self.ConditionalDiscriminator(
            self.y_real,
            self.label,
            layers=self.layers,
            channel=self.channel,
            filter_size=self.filter_size,
        )
        self.d_fake = self.ConditionalDiscriminator(
            self.y_fake,
            self.label,
            layers=self.layers,
            channel=self.channel,
            filter_size=self.filter_size,
            reuse=True,
        )

        # 損失関数の定義
        logging.info("[BUILDING]\tLoss Function")
        self.d_loss_real = loss_function.cross_entropy(
            x=self.d_real, labels=tf.ones_like(self.d_real), name="d_loss_real",
        )
        self.d_loss_fake = loss_function.cross_entropy(
            x=self.d_fake, labels=tf.zeros_like(self.d_fake), name="d_loss_fake",
        )
        self.g_loss = loss_function.cross_entropy(
            x=self.d_fake, labels=tf.ones_like(self.d_fake), name="g_loss",
        )
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # 最適化関数の定義
        logging.info("[BUILDING]\tOptimizer")
        self.g_optimizer = optimizer.adam(self.g_loss, self.learn_rate, "Generator")
        self.d_optimizer = optimizer.adam(self.d_loss, self.learn_rate, "Discriminator")


    def update(self, batch_z, batch_images, batch_labels):
        _, d_loss, _, _, summary = self.sess.run(
            [self.d_optimizer, self.d_loss, self.y_fake, self.y_real, self.summary],
            feed_dict={
                self.z: batch_z,
                self.label: batch_labels,
                self.y_real: batch_images,
            },
        )
        _, g_loss = self.sess.run(
            [self.g_optimizer, self.g_loss],
            feed_dict={self.z: batch_z, self.label: batch_labels},
        )
        return {"g_loss": g_loss, "d_loss": d_loss, "summary": summary}


