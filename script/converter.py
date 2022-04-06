'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-03 15:08:25
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 09:46:54
'''

import sys
import tensorflow as tf

class FFMtoTFRecord(object):
    def __init__(self, in_path, out_path):
        self.in_path=in_path
        self.out_path=out_path

    def split_data(self, data):
        field_idx = ""
        field_sub_idx = ""
        feature_idx = ""
        feature_vals = ""

        feat_dic = {}
        label, feature = data.split(",")
        item_array = feature.split(" ")
        for item in item_array:
            item_data = item.split(":")
            field_idx = field_idx + " " + item_data[0]
            feature_idx = feature_idx + " " + item_data[1]
            feature_vals = feature_vals + " " + item_data[2]
            
            if item_data[0] not in feat_dic:
                feat_dic[item_data[0]] = 0
            else:
                feat_dic[item_data[0]] +=1

            field_sub_idx = field_sub_idx + " " + str(feat_dic[item_data[0]])
        return int(label), field_idx, field_sub_idx, feature_idx, feature_vals

    def fit(self):
        with tf.io.TFRecordWriter(self.out_path) as writer:
            with open(self.in_path, 'r') as f:
                line = f.readline()
                while line:
                    label, field_idx, field_sub_idx, feature_idx, feature_vals = self.split_data(line)
                    record_bytes = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list = tf.train.Int64List(value=[label])),
                        "field_idx" : tf.train.Feature(bytes_list = tf.train.BytesList(value=[field_idx.encode()])),
                        "field_sub_idx" : tf.train.Feature(bytes_list = tf.train.BytesList(value=[field_sub_idx.encode()])),
                        "feature_idx" : tf.train.Feature(bytes_list = tf.train.BytesList(value=[feature_idx.encode()])),
                        "feature_vals" : tf.train.Feature(bytes_list = tf.train.BytesList(value=[feature_vals.encode()]))
                    })).SerializeToString()
                    writer.write(record_bytes)
                    line=f.readline()

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3:
        print("Error: in_path and out_path are required !")
        sys.exit()
    converter = FFMtoTFRecord(args[1], args[2])
    converter.fit()
