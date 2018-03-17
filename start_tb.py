# -*- coding: utf-8 -*-

""" Launch TensorBoard """
import os
    

def launchTensorBoard(tensorBoardPath):
    os.system('tensorboard --logdir=' + tensorBoardPath)
    return

tbpath = r"C:\Users\cvrse\Desktop\dnn_model" #Insert your path to your model folder

import threading
t = threading.Thread(target=launchTensorBoard(tbpath), args=([]))
t.start()
