import argparse
import generate_vector_noadd as generate_vector
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import cv2
import numpy as np
import sys
import subprocess
import time
import os
from os import walk
from os import listdir
from os.path import isfile, join
import glob

def initialize():
    ad_path = os.getcwd()+os.sep+opt.adpath
    onlyfiles = [f for f in listdir(ad_path) if isfile(join(ad_path, f))]
    ad_url = [x for x in onlyfiles if x.endswith('mp4')]

    length = len(ad_path+os.sep)
    for i in range(len(ad_url)):
        ad_url[i] = ad_path+os.sep+ad_url[i]

    if(opt.new):
        if(os.path.exists("Signature_noadd.txt")):
            os.remove("Signature_noadd.txt")

    ad_name = []
    ad_length = []

    for ad in ad_url:
        ad_name.append(ad[length:])
        cap = cv2.VideoCapture(ad)
        ad_length.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-125)
        print("\nInitializing...%s\n" %(str(ad[length:][0:-4])))
        if(opt.new):
            generate_vector.detect(ad, 608, 5, ad_name[len(ad_name) -1])
    print(ad_name)
    return ad_name, ad_length

def rmse(arr, desc_ad):
    x = 0
    for i in range(len(arr)):
        x += pow((arr[i] - desc_ad[i]), 2)
    x = pow(x, 0.5)
    return x

def detect(desc, ad_name, ad_length):
    img_size = 608
    source, weights, half = opt.source, 'weights/coco.pt', False
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='0')
    
    # Initialize model
    model = Darknet('cfg/yolov3-spp.cfg', img_size)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=11)

        # Validate exported model
        import onnx
        model = onnx.load('weights/export.onnx')  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    dataset = LoadImages(source, img_size=img_size, half=half)

    # Get names
    names = load_classes('data/coco.names')

    # Run inference
    sliding_box = [0]*80
    arr = []
    arr_temp = [0]*80
    frame_num = 0
    skip_duration = 0
    flag = 0
    count = 0

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if(skip_duration>0):
            skip_duration -= 1
            frame_num += 1
            continue
        else:
            t1_start = time.time()

            # Get detections
            img = torch.from_numpy(img).to(device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img)[0]

            if half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, 0.3, 0.5, classes=None, agnostic=False)

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                p, s, im0 = path, '', im0s
                ih, iw = im0.shape[:2]
                iht = int((0.15 * ih))
                ihb = int(ih - (0.15*ih))
                iwl = int((0.15 * iw))
                iwr = int(iw - (0.15 * iw))
                im0 = im0[iht:ihb, iwl:iwr]

                frame_num += 1

                # print(ad_name)
                if det is not None and len(det):  
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        arr_temp[int(c)] = int(n)

                t1_end = time.time()
                t2_start = time.time()       
                
                arr.append(arr_temp)
                arr_temp = [0]*80
                flag = 0

                if(frame_num >= 125):
                    if(frame_num > 125):
                        del arr[0]
                    for i in range(len(desc)):
                        for j in range (len(desc[i])):
                            check_arr = np.array(arr[j])
                            check_desc = np.array(desc[i][j])
                            if not (check_arr == check_desc).all():
                                flag = 1
                                break
                            flag = 0
                        if(flag == 0):
                            skip_duration = ad_length[i]
                            print(ad_name[i], end=", ")
                            print(time.strftime("%H:%M:%S", time.gmtime((frame_num - 150)/25))+"\n")
                            break
                    print("Frame : ",frame_num-124, frame_num, end =" - ")
                print((t1_end -t1_start),(time.time() - t2_start), end = '\n')
            count += 1

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='Footage_AajTak.mp4')  # input file/folder, 0 for webcam
    parser.add_argument('--adpath', type = str, default ='ads')
    parser.add_argument("--new", default = True)
    opt = parser.parse_args()

    # if(os.path.exists("result.txt")):
    #     os.remove("result.txt")
    # if(os.path.exists("rmse.txt")):
    #     os.remove("rmse.txt")
    with torch.no_grad():
        ad_name, ad_length = initialize()
        desc = ''
        with open("Signature_noadd.txt", "r") as f:
            desc = f.read()

        arr = np.resize(np.fromstring(desc, dtype = 'int', sep = ' '), (len(ad_name), 125, 80))
        print("\nInitialization Complete....Now starting detection\n")
        detect(arr, ad_name, ad_length)