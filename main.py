'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-09 08:18:49
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-06-09 21:17:00
'''

import argparse
import time
from src.DeepRank import DeepRank

def parse_args():
    date = time.strftime("%Y%m%d", time.localtime()) 
    parser = argparse.ArgumentParser(description="Run Deep Rank.")
    parser.add_argument('--model_name', nargs='?', default='DeepFM', help='model name.')
    parser.add_argument('--data_path', nargs='?', default='./tfdata', help='Input data path.')
    parser.add_argument('--saved_model_path', nargs='?', default='./saved_model', help='saved model path.')
    parser.add_argument('--version', nargs='?', default=date, help='model version.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size.')
    parser.add_argument('--buffer_size', type=int, default=40000, help='buffer size.')
    parser.add_argument('--epochs', type=int, default=100, help='epochs.')
    parser.add_argument('--feature_dim', type=int, default=1000000, help='feature dimension.')
    parser.add_argument('--embedding_dim', type=int, default=16, help='embedding dimension.')
    parser.add_argument('--field_dim', type=int, default=39, help='field dimension.')
    parser.add_argument('--field_sub_dim', type=int, default=1, help='field sub dimension.')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='drop out rate')
    parser.add_argument('--l2_reg', type=float, default=0.0001, help='l2 regularizer')
    parser.add_argument('--verbose', type=int, default=2, help='print type')
    parser.add_argument('--decay_steps', type=int, default=7335, help='steps for learning rate decay')
    parser.add_argument('--decay_rate', type=float, default= 0.5, help='rate of learning rate decay')
    parser.add_argument('--hidden_units', nargs='?', default='[512, 512, 512]', help='hidden layers configuration')
    parser.add_argument('--layer_size', nargs='?', default='[400, 400]', help='CIN layers configuration')
    parser.add_argument('--num_units', type=int, default=16, help='only used in AutoInt, num of units')
    parser.add_argument('--num_heads', type=int, default=2, help='only used in AutoInt, num of heads')
    parser.add_argument('--num_atten', type=int, default=2, help='only used in AutoInt, num of attention layers')
    parser.add_argument('--num_cross', type=int, default=4, help='only used in DeepCross, DeepCrossV2 DeepCrossMix, num of cross layers')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    DR = DeepRank(args)
    DR.train_model()
