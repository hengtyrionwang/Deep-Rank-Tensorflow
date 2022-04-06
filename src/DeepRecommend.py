'''
Descripttion:
version:
Author: Heng Tyrion Wang
Date: 2022-03-02 16:28:27
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 09:50:24
'''

import sys
import tensorflow as tf
from models.LR import LR
from models.FM import FM
from models.AFM import AFM
from models.NFM import NFM
from models.AutInt import AutoInt
from models.DeepFM import DeepFM
from models.DeepCross import DeepCross
from models.FiBiNET import FiBiNET
from models.xDeepFM import xDeepFM
from models.DeepCrossV2 import DeepCrossV2
from models.DeepCrossMix import DeepCrossMix

from utils.utils import decode


class DeepRecommend(object):
    def __init__(self, options):
        self.options = options

    def get_model(self):
        if self.options.model_name == "LR":
            model = LR(self.options)
        elif self.options.model_name == "FM":
            model = FM(self.options)
        elif self.options.model_name == "AFM":
            model = AFM(self.options)
        elif self.options.model_name == "NFM":
            model = NFM(self.options)
        elif self.options.model_name == "AutoInt":
            model = AutoInt(self.options)
        elif self.options.model_name == "DeepFM":
            model = DeepFM(self.options)
        elif self.options.model_name == "DeepCross":
            model = DeepCross(self.options)
        elif self.options.model_name == "FiBiNET":
            model = FiBiNET(self.options)
        elif self.options.model_name == "DeepCrossV2":
            model = DeepCrossV2(self.options)
        elif self.options.model_name == "DeepCrossMix":
            model = DeepCrossMix(self.options)
        elif self.options.model_name == "xDeepFM":
            model = xDeepFM(self.options)
        else:
            model = None
        return model

    def get_data(self, data_type):

        file_path = self.options.data_path + "/" + data_type + ".tfrecord"
        data_set = tf.data.TFRecordDataset(file_path)
        data_set = data_set.map(decode)
        data_set = data_set.shuffle(buffer_size = self.options.buffer_size)
        data_set = data_set.batch(batch_size = self.options.batch_size)
        return data_set

    def train_model(self):
        train_data = self.get_data("train")
        valid_data = self.get_data("valid")
        test_data = self.get_data("test")
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.options.learning_rate, decay_steps=3000, decay_rate=0.5)
        model = self.get_model()

        if model == None:
            print("This model is not availabel now!")
            return
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=[tf.keras.metrics.AUC()])
        model.fit(train_data, epochs = self.options.epochs, verbose=2, validation_data=valid_data, callbacks=[callback])
        model.evaluate(test_data, verbose=2)
        model.save(self.options.saved_model_path + "/" + self.options.model_name + "/" + self.options.version)

