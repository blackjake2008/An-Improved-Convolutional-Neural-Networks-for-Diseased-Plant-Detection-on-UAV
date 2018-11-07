#!coding=utf-8
import numpy as np
import caffe
import datetime
import pdb
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import cv2
class ssd_class(object):
    def __init__(self, 
                ssd_model_path = '../../models/VGGNet/boshi/SSD_300x300/VGG_boshi_SSD_300x300_iter_300000.caffemodel',
                ssd_deploy_path = '../../models/VGGNet/boshi/SSD_300x300/deploy.prototxt',
                ssd_mean_path = './imagenet_mean.npy',
                labelmap = "../../labelmap_boshi.prototxt",
                gpu = True,
                gpu_device = 1):
        if(gpu):
            caffe.set_mode_gpu()
            #caffe.set_device(gpu_device)  #must commented, or it would cause error
        else:
            caffe.set_mode_cpu()
        
        #self.ssd_model_path = ssd_model_path
        #self.ssd_deploy_path = ssd_deploy_path
        self.ssd_mean = np.load(ssd_mean_path).mean(1).mean(1)  # this is a tuple type
        self.labelmap = labelmap
        self.ssdnet = caffe.Classifier(ssd_deploy_path, ssd_model_path,
                mean=self.ssd_mean,
                channel_swap=(2,1,0),
                raw_scale=255,
                image_dims=(300,300))
        

    def ssd_detect(self, image, img_path):
        try:
#            pdb.set_trace()
            transformer = caffe.io.Transformer({'data': self.ssdnet.blobs['data'].data.shape})
            transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
            transformer.set_mean('data', np.array(self.ssd_mean))  # subtract the dataset-mean value in each channel
            transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
            transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
            #5. Run the net and examine the top_k results
            transformed_image = transformer.preprocess('data', image)
            self.ssdnet.blobs['data'].data[...] = transformed_image

            # Forward pass.
            detections = self.ssdnet.forward()['detection_out']
            # Parse the outputs.
            det_label = detections[0,0,:,1]
            det_conf = detections[0,0,:,2]
            det_xmin = detections[0,0,:,3]
            det_ymin = detections[0,0,:,4]
            det_xmax = detections[0,0,:,5]
            det_ymax = detections[0,0,:,6]
            imgfile=img_path.split('/')[-1]     
            result_file='../result/'+imgfile.split('/')[-1].replace('JPG','txt')
            fp=open(result_file,'w')
            CLASSES = ('background','serious','mid')
            mat=cv2.imread(img_path)
#            w=mat.shape[0]
            w=mat.shape[1]
#            h=mat.shape[1]
            h=mat.shape[0]
            for i in range(len(det_label)):
                if det_conf[i]==-1:
                   print conf[i]
                   continue
                det_xmin[i]=max(0,det_xmin[i])
                det_ymin[i]=max(0,det_ymin[i])
                #pdb.set_trace()
                print '%s %s %s %s %s %s\n'%(CLASSES[int(det_label[i])],det_conf[i],int(det_xmin[i]*w),int(det_ymin[i]*h),int(det_xmax[i]*w),int(det_ymax[i]*h))
                fp.write('%s %s %s %s %s %s\n'%(CLASSES[int(det_label[i])],det_conf[i],int(det_xmin[i]*w),int(det_ymin[i]*h),int(det_xmax[i]*w),int(det_ymax[i]*h)))
            fp.close()
        except Exception, e:
            #app_logger.error('there is no car detected! %s'%e.message)
            return "", "", "", ""


if __name__=='__main__':
    ssd_obj = ssd_class()  #initial an object
#    try:
        #  Absolute Path
    for fi in os.listdir("/sata1/yangshuang/mAP-master/images"):
        try:
            image = caffe.io.load_image('/sata1/yangshuang/mAP-master/images/%s'%fi)
        except IOError:
            continue
        print fi      
        ssd_obj.ssd_detect(image, "/sata1/yangshuang/mAP-master/images/%s"%fi)
