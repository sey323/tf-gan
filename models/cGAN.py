# import os, sys
# import time
# import cv2
# import tensorflow as tf
# import numpy as np
# import json
# from datetime import datetime
#
# sys.path.append("./common")
# import layer
# import activation
# import loss_function
# import imutil
#
# sys.path.append("./models")
# from gan import *
#
#
# class cGAN(GAN):
#     """
#     ConditionalGANの基本クラス
#     """
#
#     def __init__(
#         self,
#         input_size,
#         label_num,
#         channel=3,
#         layers=[64, 128, 256],
#         filter_size=[5, 5],
#         drop_prob=0.5,
#         zdim=100,
#         batch_num=64,
#         learn_rate=2e-4,
#         max_epoch=100,
#         gpu_config=tf.GPUOptions(per_process_gpu_memory_fraction=1.0),
#         save_folder="results/GAN",
#     ):
#
#         self.input_size = input_size
#         self.label_num = label_num
#         self.channel = channel
#         self.layers = layers
#         self.filter_size = filter_size
#         self.drop_prob = drop_prob
#         self.zdim = zdim
#         self.batch_num = batch_num
#         self.learn_rate = learn_rate
#         self.max_epoch = max_epoch
#
#         # 保存するディレクトリの作成．
#         now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
#         self.save_folder = save_folder + "/" + now
#         if self.save_folder and not os.path.exists(
#             os.path.join(self.save_folder, "images")
#         ):
#             os.makedirs(os.path.join(self.save_folder, "images"))
#             os.makedirs(os.path.join(self.save_folder, "eval"))
#             param_file = open(self.save_folder + "/param.json", "w")
#             # パラメータをJsonファイルに保存
#             json.dump(self.__dict__, param_file, indent=2)
#
#         # GPUの設定
#         self.gpu_config = tf.ConfigProto(gpu_options=gpu_config)
#
#     def ConditionalGenerator(self, z, label, channel=3, reuse=False):
#         """
#         Generator
#         """
#         print("[NETWORK]\tNoise Conditional Generator")
#
#         with tf.variable_scope("Generator", reuse=reuse) as scope:
#             if reuse:
#                 scope.reuse_variables()
#
#             # ノイズにラベルを加算
#             h = tf.concat([z, label], axis=1, name="concat_z")
#             before_output = h
#
#             # FC層
#             with tf.variable_scope("Fc{0}".format("")) as scope:
#                 dim_h, dim_w = imutil.calcImageSize(
#                     self.input_size[0],
#                     self.input_size[1],
#                     stride=2,
#                     num=len(self.layers),
#                 )
#                 output = layer.defc(
#                     h,
#                     output=[self.batch_num, dim_h, dim_w, self.layers[-1]],
#                     i="",
#                     batch_norm=False,
#                 )
#                 before_output = output
#
#             # Deconv層
#             for i, input_shape in enumerate(reversed(self.layers)):
#                 i = len(self.layers) - i
#                 if i == 1:
#                     output_shape = self.layers[0]
#                 else:
#                     output_shape = self.layers[i - 2]
#                 dim_h, dim_w = imutil.calcImageSize(
#                     self.input_size[0], self.input_size[1], stride=2, num=i - 1
#                 )
#
#                 # deconv
#                 with tf.variable_scope("deconv_layer{0}".format(i)) as scope:
#                     if i == 1:  # 最後の出力で，カラーチャネルに変更
#                         h_deconv = layer.deconv2d(
#                             before_output,
#                             stride=2,
#                             filter_size=[
#                                 self.filter_size[0],
#                                 self.filter_size[1],
#                                 input_shape,
#                                 3,
#                             ],
#                             output_shape=[self.batch_num, dim_h, dim_w, channel],
#                             i=i,
#                             batch_norm=False,
#                         )
#                         output = activation.tanh(h_deconv)
#
#                     else:
#                         h_deconv = layer.deconv2d(
#                             before_output,
#                             stride=2,
#                             filter_size=[
#                                 self.filter_size[0],
#                                 self.filter_size[1],
#                                 input_shape,
#                                 output_shape,
#                             ],
#                             output_shape=[self.batch_num, dim_h, dim_w, output_shape],
#                             i=i,
#                             batch_norm=True,
#                         )
#                         output = activation.ReLU(h_deconv)
#
#                 before_output = output
#         return output
#
#     def ConditionalDiscriminator(self, y, label, channel=3, reuse=False):
#         """
#         Discriminator
#         """
#         print("[NETWORK]\tNoise Conditional Discriminator")
#
#         with tf.variable_scope("Discriminator", reuse=reuse) as scope:
#             if reuse:
#                 scope.reuse_variables()
#             l = tf.reshape(label, [self.batch_num, 1, 1, self.label_num])
#             k = tf.ones(
#                 [self.batch_num, self.input_size[0], self.input_size[1], self.label_num]
#             )
#             k = k * l
#             h = tf.concat([y, k], axis=3)
#
#             for i, output_shape in enumerate(self.layers, 1):
#                 if i == 1:  # 1層目の時だけ
#                     before_output = h
#                     input_shape = channel + self.label_num
#
#                 # conv
#                 with tf.variable_scope("conv_layer{0}".format(i)) as scope:
#                     h_conv = layer.conv2d(
#                         x=before_output,
#                         stride=2,
#                         filter_size=[
#                             self.filter_size[0],
#                             self.filter_size[1],
#                             input_shape,
#                             output_shape,
#                         ],
#                         i=i,
#                         batch_norm=False,
#                     )
#                     output = activation.leakyReLU(h_conv)
#
#                 input_shape = output_shape
#                 before_output = output
#
#             # fc1
#             output = layer.flatten(output, self.batch_num)
#             h_fc_l = layer.fc(output, 1, 1)
#
#         return h_fc_l
#
#     def build_model(self):
#         """
#         ネットワークの全体を作成する
#         """
#
#         """変数の定義"""
#         self.z = tf.placeholder(tf.float32, [None, self.zdim], name="z")
#         self.label = tf.placeholder(tf.float32, [None, self.label_num], name="label")
#         self.y_real = tf.placeholder(
#             tf.float32,
#             [None, self.input_size[0], self.input_size[1], self.channel],
#             name="image",
#         )
#
#         """Generatorのネットワークの構築"""
#         print("[BUILDING]\tGenerator")
#         self.y_fake = self.ConditionalGenerator(self.z, self.label, self.channel)
#         self.y_sample = self.ConditionalGenerator(
#             self.z, self.label, self.channel, reuse=True
#         )
#
#         """Discrimnatorのネットワークの構築"""
#         print("[BUILDING]\tDiscriminator")
#         self.d_real = self.ConditionalDiscriminator(
#             self.y_real, self.label, self.channel
#         )
#         self.d_fake = self.ConditionalDiscriminator(
#             self.y_fake, self.label, self.channel, reuse=True
#         )
#
#         """損失関数の定義"""
#         print("[BUILDING]\tLoss Function")
#         self.g_loss = loss_function.cross_entropy(
#             x=self.d_fake,
#             labels=tf.ones_like(self.d_fake),
#             batch_num=self.batch_num,
#             name="g_loss_fake",
#         )
#
#         self.d_loss_real = loss_function.cross_entropy(
#             x=self.d_real,
#             labels=tf.ones_like(self.d_real),
#             batch_num=self.batch_num,
#             name="d_loss_real",
#         )
#         self.d_loss_fake = loss_function.cross_entropy(
#             x=self.d_fake,
#             labels=tf.zeros_like(self.d_fake),
#             batch_num=self.batch_num,
#             name="d_loss_fake",
#         )
#         self.d_loss = self.d_loss_real + self.d_loss_fake
#
#         """最適化関数の定義"""
#         print("[BUILDING]\tOptimizer")
#         self.g_optimizer = tf.train.AdamOptimizer(self.learn_rate, beta1=0.5).minimize(
#             self.g_loss,
#             var_list=[x for x in tf.trainable_variables() if "Generator" in x.name],
#         )
#         self.d_optimizer = tf.train.AdamOptimizer(self.learn_rate, beta1=0.5).minimize(
#             self.d_loss,
#             var_list=[x for x in tf.trainable_variables() if "Discriminator" in x.name],
#         )
#
#         """Tensorboadに保存する設定"""
#         print("[BUILDING]\tSAVE Node")
#         tf.summary.scalar("d_loss_real", self.d_loss_real)
#         tf.summary.scalar("d_loss_fake", self.d_loss_fake)
#         tf.summary.scalar("d_loss", self.d_loss)
#         tf.summary.scalar("g_loss", self.g_loss)
#
#         """Sessionの定義"""
#         self.sess = tf.Session(config=self.gpu_config)
#
#         ### saver
#         self.saver = tf.train.Saver()
#         self.summary = tf.summary.merge_all()
#         if self.save_folder:
#             self.writer = tf.summary.FileWriter(self.save_folder, self.sess.graph)
#
#     def train(self, batch_o):
#         self.build_model()
#         initOP = tf.global_variables_initializer()
#         self.sess.run(initOP)
#
#         # 画像を生成するラベルの作成．
#         self.test_code = []
#         for i in range(self.batch_num):
#             tmp = np.zeros(self.label_num)
#             tmp.put(i % self.label_num, 1)
#             self.test_code.append(tmp)
#
#         step = -1
#         time_history = []
#         start = time.time()
#         while batch_o.getEpoch() < self.max_epoch:
#             step += 1
#
#             batch_images, batch_labels = batch_o.getBatch(self.batch_num)
#
#             batch_z = np.random.uniform(-1.0, +1.0, [self.batch_num, self.zdim]).astype(
#                 np.float32
#             )
#             # update generator
#             _, d_loss, y_fake, y_real, summary = self.sess.run(
#                 [self.d_optimizer, self.d_loss, self.y_fake, self.y_real, self.summary],
#                 feed_dict={
#                     self.z: batch_z,
#                     self.label: batch_labels,
#                     self.y_real: batch_images,
#                 },
#             )
#             _, g_loss = self.sess.run(
#                 [self.g_optimizer, self.g_loss],
#                 feed_dict={self.z: batch_z, self.label: batch_labels},
#             )
#
#             if step > 0 and step % 10 == 0:
#                 self.writer.add_summary(summary, step)
#
#             if step % 100 == 0:
#                 train_time = time.time() - start
#                 time_history.append({"step": step, "time": train_time})
#                 print(
#                     "%6d: loss(D)=%.4e, loss(G)=%.4e; time/step = %.2f sec"
#                     % (step, d_loss, g_loss, train_time)
#                 )
#
#                 z1 = np.random.uniform(-1, +1, [self.batch_num, self.zdim])
#                 z2 = np.random.uniform(-1, +1, [self.zdim])
#                 z2 = np.expand_dims(z2, axis=0)
#                 z2 = np.repeat(z2, repeats=self.batch_num, axis=0)
#
#                 g_image1 = self.sess.run(
#                     self.y_sample, feed_dict={self.z: z1, self.label: self.test_code}
#                 )
#                 g_image2 = self.sess.run(
#                     self.y_sample, feed_dict={self.z: z2, self.label: self.test_code}
#                 )
#                 cv2.imwrite(
#                     os.path.join(self.save_folder, "images", "img_%d_real.png" % step),
#                     imutil.tileImage(y_real, size=self.label_num) * 255.0 + 128.0,
#                 )
#                 cv2.imwrite(
#                     os.path.join(self.save_folder, "images", "img_%d_fake1.png" % step),
#                     imutil.tileImage(g_image1, size=self.label_num) * 255.0 + 128.0,
#                 )
#                 cv2.imwrite(
#                     os.path.join(self.save_folder, "images", "img_%d_fake2.png" % step),
#                     imutil.tileImage(g_image2, size=self.label_num) * 255.0 + 128.0,
#                 )
#                 self.saver.save(
#                     self.sess, os.path.join(self.save_folder, "model.ckpt"), step
#                 )
#                 start = time.time()
