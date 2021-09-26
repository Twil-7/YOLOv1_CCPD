import read_data_path as rp
import numpy as np
import cv2
import train as tr
from train import SequenceData
import yolo_loss
import tiny_yolov1_model
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":

    train_x, train_y, test_x, test_y = rp.make_data()
    train_generator = SequenceData(train_x, train_y, 32)
    validation_generator = SequenceData(test_x, test_y, 32)

    # tr.train_network(train_generator, validation_generator, epoch=100)
    tr.load_network_then_train(train_generator, validation_generator, epoch=50,
                               input_name='first_weights.hdf5',
                               output_name='second_weights.hdf5')

