# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:42:03 2018

@author: cvrse
"""
""" Launch TensorBoard """
import os
    

def launchTensorBoard(tensorBoardPath):
    os.system('tensorboard --logdir=' + tensorBoardPath)
    return

tbpath = r"C:\Users\cvrse\MyDocs\PythonCodes\GasPredictor"

import threading
t = threading.Thread(target=launchTensorBoard(tbpath), args=([]))
t.start()
