import os,sys
import time
import cv2
import tensorflow as tf
import numpy as np
import json
from datetime import datetime

sys.path.append('./common')
import layer
import activation
import loss_function
import imutil


'''
GANの基本クラス
'''
class GAN( object ):
    def __init__(self ,
                input_size,
                channel= 3,
                layers = [ 64 , 128 , 256],
                filter_size = [ 5 , 5 ],
                drop_prob = 0.5,
                zdim = 100,
                batch_num = 64,
                learn_rate = 2e-4,
                max_epoch = 100,
                gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=1.0) ,
                save_folder = 'results/GAN' ):

        self.input_size = input_size
        self.channel = channel
        self.layers = layers
        self.filter_size = filter_size
        self.drop_prob = drop_prob
        self.zdim = zdim
        self.batch_num = batch_num
        self.learn_rate = learn_rate
        self.max_epoch = max_epoch

        # 保存するディレクトリの作成．
        now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.save_folder = save_folder + "/" + now
        if self.save_folder and not os.path.exists(os.path.join(self.save_folder,"images")):
            os.makedirs(os.path.join(self.save_folder,"images"))
            os.makedirs(os.path.join(self.save_folder,"eval"))
            param_file = open(self.save_folder+'/param.json','w')
            # パラメータをJsonファイルに保存
            json.dump(self.__dict__,param_file,indent=2)

        # GPUの設定
        self.gpu_config = tf.ConfigProto( gpu_options = gpu_config )


    def Generator(self,z,channel=3,reuse=False):
        '''
        Generator
        ---
            Parameters
                z       :   Input Noize
                channel :   integer
                    input color channel
                reuse   :   Boolean
                    Does share param with defined netowark?
        '''
        print('[NETWORK]\nDeep Convolutional Generator')
        with tf.variable_scope("Generator", reuse=reuse) as scope:
            if reuse: scope.reuse_variables()

            # 逆FC
            with tf.variable_scope("Fc{0}".format("")) as scope:
                dim_h , dim_w = imutil.calcImageSize(self.input_size[0],self.input_size[1],stride=2,num=len(self.layers))
                # 1層目
                output = layer.defc(z,
                                    output = [self.batch_num , dim_h , dim_w , self.layers[-1] ],
                                    i="")
                before_output = output

            # 逆畳み込みによって画像を再構成する．
            for i , input_shape in enumerate(reversed(self.layers) ):
                i = len(self.layers) - i
                if i == 1:
                    output_shape = self.layers[0]
                else:
                    output_shape = self.layers[i-2]
                dim_h , dim_w = imutil.calcImageSize(self.input_size[0],self.input_size[1],stride=2, num = i-1)

                # deconv
                with tf.variable_scope("deconv_layer{0}".format(i)) as scope:
                    if i == 1: # 最後の出力で，カラーチャネルに変更
                        h_deconv        = layer.deconv2d(   before_output,
                                                            stride=2,
                                                            filter_size = [self.filter_size[0],self.filter_size[1],input_shape,3],
                                                            output_shape=[self.batch_num,dim_h,dim_w,channel],
                                                            i = i,
                                                            batch_norm = False)
                        output  = activation.tanh( h_deconv )

                    else:
                        h_deconv       = layer.deconv2d(   before_output,
                                                            stride=2,
                                                            filter_size = [self.filter_size[0],self.filter_size[1],input_shape,output_shape],
                                                            output_shape=[self.batch_num,dim_h,dim_w,output_shape],
                                                            i = i,
                                                            batch_norm = True)
                        output  = activation.ReLU( h_deconv )
                before_output = output
        return output


    def Discriminator(self,y,channel=3,reuse=False):
        '''
        Discriminator
        '''
        print('[NETWORK]\nDeep Convolutional Discriminator')
        with tf.variable_scope("Discriminator", reuse=reuse) as scope:
            if reuse: scope.reuse_variables()

            for i , output_shape in enumerate( self.layers , 1 ):
                if i == 1:# 1層目の時だけ
                    before_output = y
                    input_shape = channel
                # conv
                with tf.variable_scope("conv_layer{0}".format(i)) as scope:
                    h_conv     = layer.conv2d( x = before_output , stride=2 , filter_size = [self.filter_size[0],self.filter_size[1] , input_shape , output_shape], i = i ,batch_norm = True)
                    output     = activation.leakyReLU( h_conv )
                    if not reuse:tf.summary.histogram("conv_layer{0}".format(i) ,output)

                input_shape = output_shape
                before_output = output

            # FC層
            output = layer.flatten(before_output,self.batch_num,"Flatten")
            output = layer.fc(output,1, "Out",batch_norm = False)

        return output


    def build_model(self):
        '''
        ネットワークの全体を作成する
        '''

        '''変数の定義'''
        self.z      = tf.placeholder( tf.float32, [None, self.zdim],name="z")
        self.y_real = tf.placeholder( tf.float32, [None, self.input_size[0], self.input_size[1], 3],name="image")

        '''Generatorのネットワークの構築'''
        print('[BUILDING]\tGenerator')
        self.y_fake   = self.Generator(self.z,self.channel)
        self.y_sample = self.Generator(self.z,self.channel,reuse=True)

        '''Discrimnatorのネットワークの構築'''
        print('[BUILDING]\tDiscriminator')
        self.d_real  = self.Discriminator(self.y_real,self.channel)
        self.d_fake  = self.Discriminator(self.y_fake,self.channel,reuse=True)

        '''損失関数の定義'''
        print('[BUILDING]\tLoss Function')
        self.g_loss      = loss_function.cross_entropy( x=self.d_fake,labels=tf.ones_like (self.d_fake), batch_num = self.batch_num , name = "g_loss_fake")

        self.d_loss_real = loss_function.cross_entropy( x=self.d_real,labels=tf.ones_like (self.d_real), batch_num = self.batch_num , name = "d_loss_real")
        self.d_loss_fake = loss_function.cross_entropy( x=self.d_fake,labels=tf.zeros_like(self.d_fake), batch_num = self.batch_num , name = "d_loss_fake")
        self.d_loss      = self.d_loss_real + self.d_loss_fake

        '''最適化関数の定義'''
        print('[BUILDING]\tOptimizer')
        self.g_optimizer = tf.train.AdamOptimizer(self.learn_rate,beta1=0.5).minimize(self.g_loss, var_list=[x for x in tf.trainable_variables() if "Generator"     in x.name])
        self.d_optimizer = tf.train.AdamOptimizer(self.learn_rate,beta1=0.5).minimize(self.d_loss, var_list=[x for x in tf.trainable_variables() if "Discriminator" in x.name])

        '''Tensorboadに保存する設定'''
        print('[BUILDING]\tSAVE Node')
        tf.summary.scalar( "d_loss_real" , self.d_loss_real)
        tf.summary.scalar( "d_loss_fake" , self.d_loss_fake)
        tf.summary.scalar( "d_loss" , self.d_loss)
        tf.summary.scalar( "g_loss" , self.g_loss)

        '''Sessionの定義'''
        self.sess = tf.Session(config=self.gpu_config)

        ### saver
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.save_folder: self.writer = tf.summary.FileWriter(self.save_folder, self.sess.graph)


    def train(self, batch):
        self.build_model()
        initOP = tf.global_variables_initializer()
        self.sess.run(initOP)

        step = -1
        start = time.time()
        while batch.getEpoch()<self.max_epoch:
            step += 1

            # ランダムにバッチと画像を取得
            batch_images,batch_labels = batch.getBatch(self.batch_num)
            batch_z = np.random.uniform(-1.,+1.,[self.batch_num,self.zdim]).astype(np.float32)

            # Update Generator
            _,g_loss = self.sess.run([ self.g_optimizer , self.g_loss ] , feed_dict={ self.z:batch_z })
            # Update Discrimnator
            _,d_loss,y_fake,y_real,summary = self.sess.run([self.d_optimizer,self.d_loss,self.y_fake,self.y_real,self.summary],feed_dict={self.z:batch_z, self.y_real:batch_images})


            if step>0 and step%10==0:
                self.writer.add_summary(summary , step )

            if step%100==0:
                # 実行時間と損失冠すの出力
                train_time = time.time()-start
                print("Epoch\t%6d\tStep%6d: loss(D)=%.4e, loss(G)=%.4e; time/step = %.2f sec"%(batch.getEpoch(),step,d_loss,g_loss,train_time))

                # ノイズの作成．
                l0 = np.array([ x%10 for x in range( self.batch_num )] , dtype = np.int32)
                z1 = np.random.uniform(-1,+1,[ self.batch_num , self.zdim ])
                z2 = np.random.uniform(-1,+1,[ self.zdim ])
                z2 = np.expand_dims( z2 , axis = 0 )
                z2 = np.repeat( z2 , repeats = self.batch_num , axis = 0 )

                # 画像を作成して保存
                g_image1 = self.sess.run(self.y_sample,feed_dict={self.z:z1})
                g_image2 = self.sess.run(self.y_sample,feed_dict={self.z:z2})
                cv2.imwrite(os.path.join(self.save_folder,"images","img_%d_real.png"%step),imutil.tileImage( y_real ) * 255.+128.)
                self.create(z1 ,os.path.join(self.save_folder,"images","img_%d_fake1.png"%step))
                self.create(z2 ,os.path.join(self.save_folder,"images","img_%d_fake2.png"%step))
                self.saver.save(self.sess,os.path.join(self.save_folder,"model.ckpt"),step)

                # 時間の計測の再開
                start = time.time()


    def restore(self, file_name):
        '''
        Restore Model
        Parameters
        ---
            file_name   :   String
                Loading Model Path
        '''
        self.build_model()
        # モデルファイルが存在するかチェック
        ckpt = tf.train.get_checkpoint_state( file_name )
        if ckpt:
            print( "[LOADING]\t" + file_name)
            ckpt_name = file_name + "/"+ckpt.model_checkpoint_path.split("/")[-1]
            self.saver.restore(self.sess, ckpt_name)
            print( "[LOADING]\t" + ckpt_name + " Complete!")
        else:
            print( file_name + ' Not found')
            exit()


    def create( self , noize , save_folder ):
        '''
        学習モデルから画像の生成
        Parameters
        ---
            noize   :   [float32]
                Input Random noize
        '''
        g_image = self.sess.run(self.y_sample,feed_dict={self.z:noize})
        cv2.imwrite(save_folder,imutil.tileImage( g_image ) * 255.+128.)
