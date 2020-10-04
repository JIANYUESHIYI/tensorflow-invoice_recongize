#coding:UTF-8
import os
import cv2

import glob
import time
import imutils
from imutils import contours

import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


NUMBER_MODEL_PATH = "model_file/model_number_only/"
NUMBER_MODEL_PATH_META = NUMBER_MODEL_PATH + ".meta"

ALL_MODEL_PATH = "model_file/model_all_character/"              
ALL_MODEL_PATH_META = ALL_MODEL_PATH + ".meta"

MODEL_5_6_G_PATH = "model_file/model_5_6_G_only/"
MODEL_5_6_G_PATH_META = MODEL_5_6_G_PATH + ".meta"

MODEL_1_3_PATH = "model_file/model_1_3_only/"
MODEL_1_3_PATH_META = MODEL_1_3_PATH + ".meta"


def showpic(canvas): 
    """
    This function is used to display pictures
    """
    global file_path
    file_path = filedialog.askopenfilename()

    print(file_path)
    img = Image.open(file_path)
    main_img = img.resize((300, 400))
    bg0 = ImageTk.PhotoImage(main_img)
    canvas.configure(image=bg0)
    canvas.image = bg0


def detect_one_char(var):
    """
    This function is used to recognize a single character
    """
    bg1_open = Image.open(file_path).resize((16, 16), Image.ANTIALIAS) 
    bg1_open_gray = bg1_open.convert('L') 
    pic = np.array(bg1_open_gray).reshape(256, )

    answer = ''
    answer = startdetect_all(pic,answer)
    var.set("The character is " + answer)
    print("The character is " + answer)


def Table2dot(raw): 
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
    rows, cols = binary.shape
    #Extract rows
    scale = 40
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated_col = cv2.dilate(eroded, kernel, iterations=1)
    #Extract cols
    scale = 20
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated_row = cv2.dilate(eroded, kernel, iterations=1)
    #Superimposed
    bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)
    # table = cv2.addWeighted(dilated_col,0.5,dilated_row,0.5,0)   
    # cv2.imwrite("res.png", bitwise_and)
    # merge = cv2.add(dilated_col, dilated_row)
    # cv2.imwrite("entire_excel_contour.png", merge)
    ys, xs = np.where(bitwise_and > 0)

    #Grid coordinates
    y_point_arr = []
    x_point_arr = []
    i = 0
    sort_x_point = np.sort(xs)
    for i in range(len(sort_x_point) - 1):
        if sort_x_point[i + 1] - sort_x_point[i] > 10:
            x_point_arr.append(sort_x_point[i])
        i = i + 1
    x_point_arr.append(sort_x_point[i])

    i = 0
    sort_y_point = np.sort(ys)
    for i in range(len(sort_y_point) - 1):
        if (sort_y_point[i + 1] - sort_y_point[i] > 10):
            y_point_arr.append(sort_y_point[i])
        i = i + 1
    y_point_arr.append(sort_y_point[i])

    return x_point_arr, y_point_arr

def getKeyWord(key, category = "A"):
    """
    This function is used to select the roi area according to the key value
    """
    raw = cv2.imread(file_path)
    x_point_arr, y_point_arr = Table2dot(raw)

    if key == 1 and category == 'A':
        h = 30
        w = 215
        x_offset = 6
        y_offset = 12

        l_t_x = x_point_arr[2]+x_offset
        l_t_y = y_point_arr[1]-y_offset-h
        r_b_x = l_t_x+w
        r_b_y= l_t_y+h

        ref =raw[l_t_y:r_b_y,l_t_x:r_b_x]

    if key == 1 and category == 'B':
        h = 25
        w = 255
        x_offset = 9
        y_offset = 54

        l_t_x = x_point_arr[2]+x_offset
        l_t_y = y_point_arr[1]-y_offset-h
        r_b_x = l_t_x+w
        r_b_y= l_t_y+h

        ref =raw[l_t_y:r_b_y,l_t_x:r_b_x]

    if key == 2:
        w = 100
        h = 22
        x_offset = 109
        y_offset = 8

        l_t_x = x_point_arr[1] + x_offset
        l_t_y = y_point_arr[7] - y_offset - h
        r_b_x = l_t_x + w
        r_b_y = l_t_y + h

        ref =raw[l_t_y:r_b_y,l_t_x:r_b_x]

    if key == 3:
        w = 100
        h = 22
        x_offset = 154
        y_offset = 8

        l_t_x = x_point_arr[6] - x_offset
        l_t_y = y_point_arr[7] - y_offset - h
        r_b_x = l_t_x + w
        r_b_y = l_t_y + h

        ref =raw[l_t_y:r_b_y,l_t_x:r_b_x]

    cv2.imshow('ref', ref)
    cv2.imwrite('pic/ref.png', ref)
    return ref


def getRefCnts(ref, method, cat = 0):
    """
    Contour detection
    Prepare for character segmentation
    """
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = imutils.resize(ref, width=400)

    if cat:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        pre_ref = cv2.dilate(ref, kernel) 
        # pre_ref = cv2.GaussianBlur(dilated,(3,3),0)
    else:
        pre_ref = cv2.GaussianBlur(ref,(3,3),0)

    ref = cv2.threshold(pre_ref, 0, 255, cv2.THRESH_BINARY_INV |
                        cv2.THRESH_OTSU)[1]

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # eroded = cv2.erode(ref, kernel)  

    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
    refCnts = contours.sort_contours(refCnts, method = method)[0]
    return refCnts,ref

 


def getOneChar(c,clone,index): 
    """
    character segmentation
    Return a single character for recognition
    """
    (x, y, w, h) = cv2.boundingRect(c)
    cut_roi = clone[y: y + h, x: x + w]

    width = 16
    height = 16
    # print(h, w)
    if h>w:
        padd_lef_rig = (h-w)//2
    else:
        padd_lef_rig = (w-h)//2
    constant= cv2.copyMakeBorder(cut_roi,0,0,padd_lef_rig,padd_lef_rig,cv2.BORDER_CONSTANT,value=(0,0,0))
    resize_roi = cv2.resize(constant, (width, height)) 

    # kernel = np.ones((2, 2), np.uint8)
    # erosion = cv2.erode(resize_roi, kernel, iterations=1)

    resize_roi1 = 255 - resize_roi  
    img_gray = cv2.cvtColor(resize_roi1, cv2.COLOR_RGB2GRAY) 
    roi = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)  

    #Add white border
    constant= cv2.copyMakeBorder(roi,3,3,3,3,cv2.BORDER_CONSTANT,value=(255,255,255))
    roi = cv2.resize(constant, (width, height))  

    cv2.imwrite("character_segmentation_result/" + str(index) + ".png", roi)
   
    file_open = "character_segmentation_result/" + str(index) + ".png"
    bg1_open = Image.open(file_open).resize((16, 16), Image.ANTIALIAS)
    bg1_open_gray = bg1_open.convert('1')  
    pic = np.array(bg1_open_gray).reshape(256, )

    return pic


def startdetect_all(pic,answer):
    """
    This function is used to recognize characters or combinations of characters and numbers
    """
    graph_all = tf.Graph()
    sess_all = tf.Session(graph = graph_all)

    with sess_all.as_default(): 
        with graph_all.as_default():
            init = tf.global_variables_initializer()
            sess_all.run(init)

            saver_all = tf.train.import_meta_graph(ALL_MODEL_PATH_META) 
            saver_all.restore(sess_all, ALL_MODEL_PATH) 

            x_input = graph_all.get_tensor_by_name("x_input:0") 
            y_conv = graph_all.get_tensor_by_name("y_pred:0")  
            keep_prob = graph_all.get_tensor_by_name("keep_prob:0")  
            prediction = tf.argmax(y_conv, 1)
            
            predint = prediction.eval(feed_dict={x_input: [pic], keep_prob:1.0}, session=sess_all)  
            print(predint)

            if predint[0] == 16 or predint[0] == 6 or predint[0] == 5:
                answer = answer + startdetect_5_6_G(pic)
            # elif predint[0] == 1 or predint[0] == 2 or predint[0] == 7 or predint[0] == 3:
            #     answer = answer + startdetect_1_2_3_7(pic)
            # elif predint[0] == 1 or predint[0] == 3:
            #     answer = answer + startdetect_1_3(pic)
            elif predint[0] < 10:
                answer = answer + str(predint[0])
            else:
                answer = answer + chr(predint[0] + 55)
        
    return answer


def startdetect_only_number(pic,answer):
    """
    This function is used to recognize numbers only
    """
    graph_num = tf.Graph()
    sess_num = tf.Session(graph = graph_num)

    with sess_num.as_default(): 
        with graph_num.as_default():
            init = tf.global_variables_initializer()
            sess_num.run(init)

            saver_num = tf.train.import_meta_graph(NUMBER_MODEL_PATH_META) 
            saver_num.restore(sess_num, NUMBER_MODEL_PATH) 

            x_input = graph_num.get_tensor_by_name("x_input:0") 
            y_conv = graph_num.get_tensor_by_name("y_pred:0")  
            keep_prob = graph_num.get_tensor_by_name("keep_prob:0")  
            prediction = tf.argmax(y_conv, 1)

            predint = prediction.eval(feed_dict={x_input: [pic], keep_prob:1.0}, session=sess_num) 

            # if predint[0] == 1 or predint[0] == 2 or predint[0] == 7 or predint[0] == 3:
            #     answer = answer + startdetect_1_2_3_7(pic)
            # if predint[0] == 1 or predint[0] == 3:
            #     answer = answer + startdetect_1_3(pic)
            if predint[0] == 6 or predint[0] == 5:
                answer = answer + startdetect_5_6_G(pic)
            elif predint[0] < 10:
                answer = answer + str(predint[0])
            else:
                answer = answer + chr(predint[0] + 55)
            
    return answer



def startdetect_5_6_G(pic):
    """
    This function is used to recognize numbers only
    """
    graph_5_6_G = tf.Graph()
    sess_5_6_G = tf.Session(graph = graph_5_6_G)

    with sess_5_6_G.as_default(): 
        with graph_5_6_G.as_default():
            init = tf.global_variables_initializer()
            sess_5_6_G.run(init)

            saver_5_6_G = tf.train.import_meta_graph(MODEL_5_6_G_PATH_META) 
            saver_5_6_G.restore(sess_5_6_G, MODEL_5_6_G_PATH) 

            x_input = graph_5_6_G.get_tensor_by_name("x_input:0") 
            y_conv = graph_5_6_G.get_tensor_by_name("y_pred:0")  
            keep_prob = graph_5_6_G.get_tensor_by_name("keep_prob:0")  
            prediction = tf.argmax(y_conv, 1)

            predint = prediction.eval(feed_dict={x_input: [pic], keep_prob:1.0}, session=sess_5_6_G) 
            # print(predint[0])

            if predint[0] == 0:                      #6
                answer = str(predint[0] + 6)
            elif predint[0] == 1:                   #5
                answer = str(predint[0] + 4)
            else:                                   #G
                answer = chr(predint[0] + 69)

    return answer


# def startdetect_1_2_3_7(pic):
#     """
#     This function is used to recognize numbers only
#     """
#     graph_1_2_3_7 = tf.Graph()
#     sess_1_2_3_7 = tf.Session(graph = graph_1_2_3_7)

#     with sess_1_2_3_7.as_default(): 
#         with graph_1_2_3_7.as_default():
#             init = tf.global_variables_initializer()
#             sess_1_2_3_7.run(init)

#             saver_1237 = tf.train.import_meta_graph(MODEL_1_2_3_7_PATH_META) 
#             saver_1237.restore(sess_1_2_3_7, MODEL_1_2_3_7_PATH) 

#             x_input = graph_1_2_3_7.get_tensor_by_name("x_input:0") 
#             y_conv = graph_1_2_3_7.get_tensor_by_name("y_pred:0")  
#             keep_prob = graph_1_2_3_7.get_tensor_by_name("keep_prob:0")  
#             prediction = tf.argmax(y_conv, 1)

#             predint = prediction.eval(feed_dict={x_input: [pic], keep_prob:1.0}, session=sess_1_2_3_7) 
#             print("----")
#             print(predint[0])

#             if predint[0] == 3:
#                 answer = str(predint[0] + 4)
#             else:
#                 answer = str(predint[0] + 1)

#     return answer



# def startdetect_1_3(pic):
#     """
#     This function is used to recognize numbers only
#     """
#     graph_1_3 = tf.Graph()
#     sess_1_3 = tf.Session(graph = graph_1_3)

#     with sess_1_3.as_default(): 
#         with graph_1_3.as_default():
#             init = tf.global_variables_initializer()
#             sess_1_3.run(init)

#             saver_13 = tf.train.import_meta_graph(MODEL_1_3_PATH_META) 
#             saver_13.restore(sess_1_3, MODEL_1_3_PATH) 

#             x_input = graph_1_3.get_tensor_by_name("x_input:0") 
#             y_conv = graph_1_3.get_tensor_by_name("y_pred:0")  
#             keep_prob = graph_1_3.get_tensor_by_name("keep_prob:0")  
#             prediction = tf.argmax(y_conv, 1)

#             predint = prediction.eval(feed_dict={x_input: [pic], keep_prob:1.0}, session=sess_1_3) 
#             print("----")
#             print(predint[0])

#             if predint[0] == 0:                     #1
#                 answer = str(predint[0] + 1)
#             elif predint[0] == 1:                   #3
#                 answer = str(predint[0] + 2)

#     return answer



def batch_getKeyWord(file_path, key, category = "A"):
    """
    This function is used to get roi area in batch recognition
    """
    raw = cv2.imread(file_path)
    x_point_arr, y_point_arr = Table2dot(raw)

    if key == 1 and category == 'A':
        h = 30
        w = 215
        x_offset = 6
        y_offset = 12

        l_t_x = x_point_arr[2]+x_offset
        l_t_y = y_point_arr[1]-y_offset-h
        r_b_x = l_t_x+w
        r_b_y= l_t_y+h

        ref_in =raw[l_t_y:r_b_y,l_t_x:r_b_x]

        return ref_in

    if key == 1 and category == 'B':
        h = 25
        w = 255
        x_offset = 9
        y_offset = 54

        l_t_x = x_point_arr[2]+x_offset
        l_t_y = y_point_arr[1]-y_offset-h
        r_b_x = l_t_x+w
        r_b_y= l_t_y+h

        ref_in =raw[l_t_y:r_b_y,l_t_x:r_b_x]

        return ref_in

    if key == 2:
        w = 100
        h = 22
        x_offset = 109
        y_offset = 8

        l_t_x = x_point_arr[1] + x_offset
        l_t_y = y_point_arr[7] - y_offset - h
        r_b_x = l_t_x + w
        r_b_y = l_t_y + h

        ref_ne =raw[l_t_y:r_b_y,l_t_x:r_b_x]

        return ref_ne

    if key == 3:
        w = 100
        h = 22
        x_offset = 154
        y_offset = 8

        l_t_x = x_point_arr[6] - x_offset
        l_t_y = y_point_arr[7] - y_offset - h
        r_b_x = l_t_x + w
        r_b_y = l_t_y + h

        ref_gr =raw[l_t_y:r_b_y,l_t_x:r_b_x]

        return ref_gr
