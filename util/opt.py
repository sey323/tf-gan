#!/usr/bin/python
import json


def save_param( type , img_path, save_folder ,model_path = '', file_num = '', test_num = '' ):
    '''
    学習パラメータの保存
    '''
    import tensorflow as tf

    # データを整形
    data = {
        "type":type,
        "img_path":img_path,
        "model_path":model_path,
        "file_num":file_num,
        "test_num":test_num
    }
    param_file = open(save_folder+'/exp.json','w')
    # パラメータを保存
    json.dump( data ,param_file,indent=2)
