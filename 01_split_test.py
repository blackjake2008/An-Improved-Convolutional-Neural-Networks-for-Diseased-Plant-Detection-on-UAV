import numpy as np
from PIL import Image
import cv2
import sys
import os
import pdb



def splilt_img(src, src_w, src_h, block_w, block_h):
    x_num = int(src_w/block_w)
    y_num = int(src_h/block_h)
    x_left = src_w%block_w
    y_left = src_h%block_h
    i = 0
    j = 0
    #pdb.set_trace()
    des_list = []
    #pdb.set_trace()
    for j in range(y_num):
        for i in range(x_num):
            start_x = i*block_w
            start_y = j*block_h
            print(start_x, start_y)
            tempsrc = src[start_y:start_y+block_h, start_x:start_x+block_w]
            des_list.append(tempsrc)
        #pdb.set_trace()
        if x_left >= 0:
            i = i+1
            start_x = i*block_w
            start_y = j*block_h
            print(start_x, start_y)
            tempsrc = src[start_y:start_y+block_h, start_x:start_x+x_left]
            des_list.append(tempsrc)
#    pdb.set_trace()
    if y_left >= 0:
        for i in range(x_num):
            start_x = i*block_w
            start_y = y_num*block_h
            print(start_x, start_y)
            tempsrc = src[start_y:start_y+y_left, start_x:start_x+block_w]
            des_list.append(tempsrc)
        if x_left >= 0:
            i = i+1
            start_x = x_num*block_w
            start_y = y_num*block_h
            print(start_x, start_y)
            tempsrc = src[start_y:start_y+y_left, start_x:start_x+x_left]
            des_list.append(tempsrc)
    #pdb.set_trace()
    return des_list


if __name__=="__main__":
    Inputpath = "G:\\tmp\\tmp20\\02_whole_plot"
    outputpath = "G:\\tmp\\tmp20\\03_whole_plot_split"

    for filename in os.listdir(Inputpath):
        if filename.find('.JPG') is not -1:
            src_filepath = os.path.join(Inputpath, filename)
            #pdb.set_trace()
            src_img = cv2.imread(src_filepath)
            split_img_list = splilt_img(src_img, src_img.shape[1], src_img.shape[0], 1000, 1000)
            #pdb.set_trace()
            for i in range(len(split_img_list)):
                result_path = os.path.join(outputpath, filename)
                #pdb.set_trace()
                if i==64:
                    pdb.set_trace()
                res_path = result_path.replace(".JPG", "_%02d.JPG"%(i))
                #pdb.set_trace()
                cv2.imwrite(res_path, split_img_list[i])
