import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

def detect(source, img_size, nsecs, ad_name):
    weights, half = 'weights/coco.pt', False
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='0')

    # Initialize model
    model = Darknet('cfg/yolov3-spp.cfg', img_size)

    #Load weights
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

    # Get names and colors
    names = load_classes('data/coco.names')

    # Run inference
    t0 = time.time()
    count = 0
    arr = [0]*80
    arr_final = []
    
    for path, img, im0s, vid_cap in dataset:
        if(count==25*nsecs):
            break
        else:
            t = time.time()

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
            for i, det in enumerate(pred): # detections per image 
                
                p, s, im0 = path, '', im0s
                ih, iw = im0.shape[:2]
                iht = int((0.15 * ih))
                ihb = int(ih - (0.15*ih))
                iwl = int((0.15 * iw))
                iwr = int(iw - (0.15 * iw))
                im0 = im0[iht:ihb, iwl:iwr]
                count +=1

                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                        arr[int(c)] = int(n)
                    
                arr_final.append(arr)
                for i in range (len(arr_final[0])):
                    with open ("Signature_noadd.txt", "a+") as f:
                        f.write(str(arr_final[0][i])+" ")
                del arr_final[0]

                arr = [0]*80
                print("%s - Frame %d : " %(ad_name, count), end = "")
                print('%sDone. (%.3fs)' % (s, time.time() - t))

    print('Done. (%.3fs)\n' % (time.time() - t0))


if __name__ == '__main__':
    if(os.path.exists("Signature_noadd.txt")):
        os.remove("Signature_noadd.txt")
    
    with torch.no_grad():
        detect("ads/platinum.mp4", 608, 5, "platinum")