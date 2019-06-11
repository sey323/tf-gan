import os,sys
sys.path.append('./util')
import imload,opt
from batchgen import *

# Ganモデルの読み込み
sys.path.append('./models')
from gan import *


'''
メイン関数
'''
def main(FLAGS):
    # パラメータの取得
    img_path = os.getenv("DATASET_FOLDER", "dataset")
    save_path = os.getenv("SAVE_FOLDER", "results")


    '''設定ファイルからパラメータの読み込み'''
    print('[LOADING]\tmodel parameters loding')

    # ファイルのパラメータ
    folder = FLAGS.folder
    resize = [ FLAGS.resize , FLAGS.resize ]
    file_num = FLAGS.file_num
    gray = FLAGS.gray
    channel = 1 if gray else 3

    # GANのクラスに関するパラメータ
    layers = [64 , 128 , 256]
    max_epoch = FLAGS.max_epoch
    batch_num = FLAGS.batch_size
    save_folder = FLAGS.save_folder
    save_path = save_path + '/' + save_folder


    '''画像の読み込み'''
    train_image ,train_label = imload.make( folder , gray = gray , train_num = file_num , img_size = resize[0] )


    '''バッチの作成'''
    batch = batchgen( train_image , train_label)


    '''モデルの作成'''
    print("[LOADING]\tGAN")
    gan = GAN(           input_size=resize,
                         channel = channel,
                         layers = layers,
                         batch_num = batch_num ,
                         max_epoch = max_epoch ,
                         save_folder = save_path)

    # 学習の開始
    gan.train( batch )

    # 最後にパラメータを保存
    opt.save_param(
            type = FLAGS.type,
            folder=folder,
            file_num=file_num,
            save_folder=gan.save_folder)


if __name__=="__main__":
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # 実行するGANモデルの指定．
    flags.DEFINE_string('type', 'gan', 'Choice GAN type.')

    # 読み込む画像周り
    flags.DEFINE_string('folder', '', 'Directory to put the training data.')
    flags.DEFINE_integer('resize', 64, 'Size of Image.')
    flags.DEFINE_integer('file_num', 0, 'Loading Images Num.')
    flags.DEFINE_boolean('gray', False, 'Convert Gray Scale?')

    # GANの学習パラメータ
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('max_epoch', 100, 'Number of steps to run trainer.')
    flags.DEFINE_integer('batch_size', 25, 'Batch size.  ''Must divide evenly into the dataset sizes.')

    # 保存フォルダの決定
    flags.DEFINE_string('save_folder', '', 'Data save folder')

    main(FLAGS)
