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
net_file = '../../gen_mobilenet/example/MobileNetSSD_deploy.prototxt'
#caffe_model='/sata1/liangdas_ssd/test_mobile/test_common_moblie/model/mobilenet_iter_105000.caffemodel'
caffe_model='../../gen_mobilenet/example/MobileNetSSD_deploy.caffemodel'
test_dir = "/sata1/liangdas_ssd/2_50mm_split"
result_dir = "../result/"
if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
caffe.set_mode_gpu()
caffe.set_device(3)
net = caffe.Net(net_file,caffe_model,caffe.TEST)  
CLASSES = ('background','serious','mid')
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
    result_file='../result/'+imgfile.split('/')[-1].replace('JPG','txt')
    fp=open(result_file,'w')
    for i in range(len(box)):
        if conf[i]==-1:
            print conf[i]
            continue
        print '%s %s %s %s %s %s\n'%(CLASSES[int(cls[i])],conf[i],box[i][0],box[i][1],box[i][2],box[i][3])
        fp.write('%s %s %s %s %s %s\n'%(CLASSES[int(cls[i])],conf[i],box[i][0],box[i][1],box[i][2],box[i][3]))
    fp.close()
        #top_conf.append(conf[i])
        #top_xmin.append(box[i][0])
        #top_ymin.append(box[i][1])
        #top_xmax.append(box[i][2])
        #top_ymax.append(box[i][3])
        #top_labels.append(CLASSES[int(cls[i])])
        #image = caffe.io.load_image(imgfile)
        #if len(top_labels)>0:
           #create_xml(imgfile, image, top_conf, top_labels, top_xmin, top_xmax, top_ymin, top_ymax,threshold)
           # p1 = (box[i][0], box[i][1])
           # p2 = (box[i][2], box[i][3])
           # cv2.rectangle(origimg, p1, p2, (0,255,0),5)
           # p3 = (max(p1[0], 15), max(p1[1], 15))
           # title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       # cv2.putText(origimg, title, p3, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
       # #pdb.set_trace()
       # cv2.imwrite(result_dir+imgfile.split('/')[-1],origimg)
       # print result_dir+imgfile.split('/')[-1]
#cv2.imshow("SSD", origimg)
 
#k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
#if k == 27 : return False
#return True
for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break
