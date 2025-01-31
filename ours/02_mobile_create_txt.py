import numpy as np  
import sys,os  
import cv2
import pdb
import caffe
import datetime
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from google.protobuf import text_format
from caffe.proto import caffe_pb2
caffe_root = '/home/soft/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')

#net_file= '/sata1/liangdas_ssd/test_mobile/test_common_moblie/model/MobileNetSSD_deploy.prototxt'  
net_file = '../../my_MobileNetSSDV2_deploy.prototxt'
#caffe_model='/sata1/liangdas_ssd/test_mobile/test_common_moblie/model/mobilenet_iter_105000.caffemodel'
caffe_model='../../snapshot/mobilenet_iter_300000.caffemodel'
#test_dir = "../../../2_50mm_split"
test_dir = "../../../03_2_50mm_height"
result_dir = "../result_height/"
if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(net_file,caffe_model,caffe.TEST)  
CLASSES = ('background','serious','mid')
#CLASSES = ('background','dead','infected')
def get_labelname(labelmap_file, labels):
#pdb.set_trace()
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
    for i in xrange(0, num_labels):
        if label == labelmap.item[i].label:
            found = True
            labelnames.append(labelmap.item[i].display_name)
            break
        assert found == True
    return labelnames

def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    for b in box:
        for i in range(4):
            if b[i] < 0: 
                b[i]=0
        if b[0] > w:
           b[0] = w
        if b[2] > w:
           b[2] = w
        if b[1] > h:
           b[1] = h
        if b[3] > h:
           b[3] = h
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    print imgfile
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)
    result_file=os.path.join(result_dir, imgfile.split('/')[-1].replace('JPG','txt'))
    fp=open(result_file,'w')
    for i in range(len(box)):
        if conf[i]==-1:
            print conf[i]
            continue
        print '%s %s %s %s %s %s\n'%(CLASSES[int(cls[i])],conf[i],box[i][0],box[i][1],box[i][2],box[i][3])
        fp.write('%s %s %s %s %s %s\n'%(CLASSES[int(cls[i])],conf[i],box[i][0],box[i][1],box[i][2],box[i][3]))
    fp.close()


for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break
