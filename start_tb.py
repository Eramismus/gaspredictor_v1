# -*- coding: utf-8 -*-

""" Launch TensorBoard """
import os
    

def launchTensorBoard(tensorBoardPath):
    os.system('tensorboard --logdir=' + tensorBoardPath)
    return

tbpath = r"Directory\to\your\model\folder" #Insert your path to your model folder

import threading
t = threading.Thread(target=launchTensorBoard(tbpath), args=([]))
t.start()
