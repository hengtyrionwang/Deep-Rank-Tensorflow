'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-12 08:33:04
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 10:24:47
'''

import sys
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

def read_line(path):
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            yield line
            line=f.readline()

def split_data(data):
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

def get_batch_data(path, batch_size):
    batched_field_idx = []
    batched_field_sub_idx = []
    batched_feature_idx = []
    batched_feature_vals = []
    reader = read_line(path)
    for i in range(batch_size):
        line = next(reader)
        label, field_idx, field_sub_idx, feature_idx, feature_vals = split_data(line)
        batched_field_idx.append([field_idx])
        batched_field_sub_idx.append([field_sub_idx])
        batched_feature_idx.append([feature_idx])
        batched_feature_vals.append([feature_vals])
    return batched_field_idx, batched_field_sub_idx, batched_feature_idx, batched_feature_vals

def predict(path, batch_size, model_name):

    field_idx, field_sub_idx, feature_idx, feature_vals = get_batch_data(path, batch_size)
    channel = grpc.insecure_channel('localhost:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()

    request.model_spec.name = model_name
    request.model_spec.signature_name = 'serving_default' 
    request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(field_idx))
    request.inputs['input_2'].CopyFrom(tf.make_tensor_proto(field_sub_idx)) 
    request.inputs['input_3'].CopyFrom(tf.make_tensor_proto(feature_idx)) 
    request.inputs['input_4'].CopyFrom(tf.make_tensor_proto(feature_vals))  

    result = stub.Predict(request) 
    
    return result

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 4:
        print("Error: path, batch_size and model_name are required !")
        sys.exit()
    results = predict(args[1], int(args[2]), args[3])
    print(results)
