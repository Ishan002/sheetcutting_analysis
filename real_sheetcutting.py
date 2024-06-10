"""
Version 1.3
Sheet-cutting Machine, New Script with Model of 4 classes - Hook, Spark, Plate, Lifter
updates:
- Updated method for Plate inside/outside ROI for Idle/Inactive condition
- Changes made in Time_keeper method
RUN COMMAND:  python SheetCutting_Tracker.V1.1_Main.py --source rtsp://admin:Pass@1234@10.7.157.192/Streaming/Channels/101 --cam cam *(Camera number ex: BSCAM6)
"""

import datetime
import math
from email.mime.text import MIMEText
from pathlib import Path
import cv2, argparse
import torch
# import  mysql.connector as sql
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import  non_max_suppression,scale_coords
from utils.torch_utils import select_device, TracedModel
from VideoRecorder import videosave
from customsort import *
import time
import DataBase_generator_14sept
dbdata = DataBase_generator_14sept.dbconnect("localhost", "root",'', "sheetcutting_db", "machine_monitoring")


plate_window, hook_window, spark_window, loading_or_unloading,lifter_window = [], [], [], [], []
active, inactive, loading, unloading = 0, 0, 0, 0
old_machine_status, machine_status, dryrunFlag = None, "Idle", None
start_time, status_time, dryrun_starttime = 0, 0, 0
Total_Active, Total_Inactive, Total_Unloading, Total_Loading = 0,0,0,0
sort_tracker = None
Machine_status_history = []


def tracker_initialize():
    global sort_tracker
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    return sort_tracker

def LoadModel(weights):
    device = select_device('0')
    half = True  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = 640  # check img_size
    if half:
        model.half()  # to FP16
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    return model, names, colors, device

def image_preprocess(img, device):
    img = letterbox(img, 640, 32)[0]
        # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def get_machine_status(dets):
    global plate_window, hook_window, spark_window,lifter_window, sort_tracker
    global machine_status
    if len(dets) == 0:
        hook_window.append(0) 
        spark_window.append(0)
        plate_window.append(0)
        lifter_window.append(0)
        #--------------------- SETTING UP WINDOW SIZE ----------------------#
        if len(plate_window)>250:
            plate_window.pop(0)
        if len(hook_window)>100:
            hook_window.pop(0)
        if len(spark_window)>100:
            spark_window.pop(0)
        if len(lifter_window)>=300:
            #print(lifter_window,"\n",len(lifter_window))
            lifter_window.pop(0)
        #-------------------------------------------------------------------#
        if plate_window.count(0)>70:
            machine_status = "Idle"
        if plate_window.count(-1)>10:
            if hook_window.count(1)>10:
                machine_status = "Loading"
        if plate_window.count(1)>10:
            machine_status = "Inactive"
        if lifter_window.count(1)>10:
            #print("LIFTER IN ROI: ",lifter_window.count(1))
            machine_status = "Unloading"
        #----- Overriding all operations with Active if Spark is seen -----#
        if spark_window.count(1)>=10:
            machine_status = "Active"

    else:
        #print("get machine status:", datetime.datetime.now())
        [hook_window.append(1) if len((np.where(dets[:,4] == 0))[0])!=0 else hook_window.append(0)]
        [spark_window.append(1) if len((np.where(dets[:,4] == 1))[0])!=0 else spark_window.append(0)]
        if len((np.where(dets[:,4] == 2))[0])!=0:
            r, _ = ROI_Checker(dets, 2)
            [plate_window.append(1) if r==1 else plate_window.append(-1)]
        if len((np.where(dets[:,4] == 2))[0])==0:
            plate_window.append(0) 
        if len((np.where(dets[:,4] == 3))[0])!=0:
            r, _ = ROI_Checker(dets, 3)
            [lifter_window.append(1) if r==1 else lifter_window.append(0)]
        elif len((np.where(dets[:,4] == 3))[0]) == 0:
            #print("no lifter found")
            #print(time.time())
            lifter_window.append(0)
        
        #--------------------- SETTING UP WINDOW SIZE ----------------------#
        if len(plate_window)>500:
            plate_window.pop(0)
        if len(hook_window)>100:
            hook_window.pop(0)
        if len(spark_window)>100:
            spark_window.pop(0)
        if len(lifter_window)>=300:
            #print(lifter_window,"\n",len(lifter_window))
            lifter_window.pop(0)
        #-------------------------------------------------------------------#

        if plate_window.count(0)>70:
            machine_status = "Idle"
        if plate_window.count(-1)>200:
            if hook_window.count(1)>10:
                machine_status = "Loading"
        if plate_window.count(1)>10:
            machine_status = "Inactive"
        if lifter_window.count(1)>10:
            #print("LIFTER IN ROI: ",lifter_window.count(1))
            machine_status = "Unloading"
        #----- Overriding all operations with Active if Spark is seen -----#
        if spark_window.count(1)>=10:
            #print("time: ", datetime.datetime.now(), " ",spark_window.count(1))
            machine_status = "Active"

def time_keeper():
    global old_machine_status, status_time, start_time, Machine_status_history
    mach_stat = {"Idle":0,"Active":1,"Inactive":2,"Loading":3}
    if old_machine_status == machine_status:
        secs = int(time.time()-start_time)
        status_time = str(round((secs/60),2))
        if secs > 3600:
            dbdata.add_dbdata(["Hazira", "Sheetcutting", f"{camname}", old_machine_status , status_time])
            start_time = time.time()
        if "23:59:59"<datetime.datetime.now().strftime("%H:%M:%S")<"23:59:59":
            dbdata.add_dbdata(["Hazira", "Sheetcutting", f"{camname}", old_machine_status , status_time])
            start_time = time.time()
    else:
        secs = int(time.time()-start_time)
        status_time = round((secs/60),2)
        print(f"{old_machine_status}: {status_time}")
        print(f"Machine past Operation: {Machine_status_history}")
        if old_machine_status == "Inactive" and "Active" in Machine_status_history:
            if status_time < 2.0:
                old_machine_status = "Dry Run"
        #Push data into Database with last old_machine_status as machine status and tdh as Total time taken by process
        if old_machine_status != None:
            pass
            dbdata.add_dbdata(["Hazira", "Sheetcutting", f"{camname}", old_machine_status , status_time])
        start_time = time.time()
        old_machine_status = machine_status
        if len(Machine_status_history)>2:
            Machine_status_history.pop(0)
        Machine_status_history.append(old_machine_status) 

def ROI_Checker(det_arr, dobj, _del = False):
    delrows =[]
    rflag = 0
    if camname == "BSCAM6":
        area = [(139,139),(661,139),(1881,665),(750, 889)]
    if camname == "BSCAM7":  
        area =  [(277,591),(1165,321),(1519,383),(1081,745),(277,591)]
    if camname == "BSCAM8":
        #area =[(1351,193),(1697,193),(1351,1079),(10,1079),(10,913)]          #EXTENDED
        area = [(730,513),(1553,95),(1733,93),(1511,587),(721,515)]            #FULLBED
        #area = [(1319,205),(1651,223),(1451,569),(753,501)]   #ORIGINAL
    if (np.where(det_arr[:,4] == dobj)[0]).size!=0:
        for i in range(len(np.where(det_arr[:,4] == dobj)[0])):
            #print("Detected class: ",np.where(det_arr[:,4] == dobj)[0])
            x1,y1,x2,y2,_,_,_,_,_ = det_arr[np.where(det_arr[:,4] == dobj)[0][i]]
            centerpoint = (int((x1+x2)/2),int((y1+y2)/2))
            r = cv2.pointPolygonTest(np.array(area, np.int32), centerpoint, False)
            if r == 1.0:
                rflag = 1
           
    return rflag, det_arr

def trackerr(im0):
    global machine_status
    tracks = sort_tracker.getTrackers()
    #print(tracks.unpack())
    for i,track in enumerate(tracks):
        #print(track.detclass[-1])
        try:
            dist = math.dist(track.centroidarr[-1], track.centroidarr[-20])
            print(dist)
            if track.centroidarr[-1][0] > track.centroidarr[-20][0] and dist> 100:
                cv2.line(im0, (int(track.centroidarr[-1][0]),
                                                            int(track.centroidarr[-1][1])),
                                                            (int(track.centroidarr[-2][0]),
                                                                int(track.centroidarr[-2][1])),
                                                            (50, 255, 255), thickness=2)
                                                            
                cv2.putText(im0, str(dist), (int(track.centroidarr[-1][0]-10),int(track.centroidarr[-1][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2)
                cv2.putText(im0, "Moving Right",(int(track.centroidarr[-1][0]),int(track.centroidarr[-1][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2)
                loading_or_unloading.append(1)
            if track.centroidarr[-1][0] < track.centroidarr[-20][0] and dist> 100:
                cv2.line(im0, (int(track.centroidarr[-1][0]),
                                                            int(track.centroidarr[-1][1])),
                                                            (int(track.centroidarr[-2][0]),
                                                                int(track.centroidarr[-2][1])),
                                                            (50, 50, 255), thickness=2)
                cv2.putText(im0, str(dist), (int(track.centroidarr[-1][0]-10),int(track.centroidarr[-1][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
                cv2.putText(im0, "Moving Left", (int(track.centroidarr[-1][0]),int(track.centroidarr[-1][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
                loading_or_unloading.append(-1)
            else:
                loading_or_unloading.append(0)
        except:
            pass
    return im0

def get_dryrun_status(det_arr):
    global dryrunFlag, dryrun_starttime
    if machine_status == "Active":
        if (np.where(det_arr[:,4] == 1)[0]).size == 0:
            if dryrunFlag != 0:
                secs = int(time.time() - dryrun_starttime)
                #print(f"Seconds: {secs}")
                dryrun = str(round((secs/60),2))
            else:
                dryrun_starttime = time.time()
                dryrunFlag = 1
        else:
            dryrunFlag = 0
    elif old_machine_status!=machine_status:
        secs = int(time.time()-dryrun_starttime)
        dryrun = round((secs/60),2)
        print(f"Last DryRun Time: {dryrun}")

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0), rs=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        label = names[cat]
        colorcode=[(80,255, 30),(86, 86, 255), (80,255, 30), (40,255,249),  (86, 86, 253), (40,255,105)]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if cat!=0 or cat!=1 or cat!=2:
            cv2.rectangle(img, (x1, y1), (x2, y2), colorcode[cat], 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w + 10, y1), colorcode[cat], -1)
            cv2.putText(img, f'{id}:{label}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 0], 2)
    return img

def annotations(pred, img, im0, sort_tracker, names):
    global machine_status
    if camname == "BSCAM6":
        area = [(139,139),(661,139),(1881,665),(750, 889)]
    if camname == "BSCAM7":  
        area =  [(277,591),(1165,321),(1519,383),(1081,745),(277,591)]
    if camname == "BSCAM8":
        #area =[(1351,193),(1697,193),(1351,1079),(10,1079),(10,913)]          #EXTENDED
        area = [(730,513),(1553,95),(1733,93),(1511,587),(721,515)]            #FULLBED
        #area = [(1319,205),(1651,223),(1451,569),(753,501)] 
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        overlay = im0.copy()
        cv2.rectangle(overlay, (20,170),(580, 210), (0, 0, 0), -1)
        cv2.rectangle(im0, (20,170),(580, 210), (0, 255, 255), 1)
        alpha = 0.8 # Transparency factor.
        im0 = cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0)
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            dets_to_sort = np.empty((0, 6))
            for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                if detclass == 1:
                    r = cv2.pointPolygonTest(np.array(area, np.int32),(int((x1+x2)/2),int((y1+y2)/2)), False)
                    #print("R Value: ",r)
                    if r == 1.0:
                        dets_to_sort = np.vstack((dets_to_sort,
                                                    np.array([x1, y1, x2, y2, conf, detclass])))
                    elif r == -1.0: 
                        pass
                else:
                    dets_to_sort = np.vstack((dets_to_sort,
                                                        np.array([x1, y1, x2, y2, conf, detclass])))
            tracked_dets = sort_tracker.update(dets_to_sort)
            #_, tracked_dets = ROI_Checker(tracked_dets, int(1), _del =True)         #spark
            get_machine_status(tracked_dets)
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                im0 = draw_boxes(im0, bbox_xyxy, identities, categories, names)
        else:
            get_machine_status(det)
   
    return im0
# ..............................................................................

def detect(source, Camname):
    global status_time
    source, weights = source, "2.pt"
    sort_tracker = tracker_initialize()
    # filename = source.split('.')[0]
    vidsave = videosave(source, f"ttemp.mp4")
    model, names, color, device = LoadModel(weights)
    cap = cv2.VideoCapture(source)
    framenum = 0
    if camname == "BSCAM6":
        area = [(139,139),(661,139),(1881,665),(750, 889)]
    if camname == "BSCAM7":  
        area =  [(277,591),(1165,321),(1519,383),(1081,745),(277,591)]
    if camname == "BSCAM8":
        #area =[(1351,193),(1697,193),(1351,1079),(10,1079),(10,913)]          #EXTENDED
        area = [(730,513),(1553,95),(1733,93),(1511,587),(721,515)]            #FULLBED
        #area = [(1319,205),(1651,223),(1451,569),(753,501)]   #ORIGINAL
    while True:
        ret, img = cap.read()
        #cv2.imwrite(f'{camname}.jpg',img)
        if ret:
            im0 = img
            framenum += 1
            if framenum % 1.5 ==0: 
                img = image_preprocess(img, device)
                # Inference
                pred = model(img)[0]
                mach_stat = {"Idle":0,"Active":1,"Inactive":2,"Loading":3}
                # Apply NMS
                pred = non_max_suppression(pred, 0.30, 0.20)  
                im0  = annotations(pred, img, im0, sort_tracker, names)
                state = {"Active":[20,255,10],"Inactive":[0,240,240],"Loading":[0,200,255],"Unloading":[0,255,200],"Idle":[150,150,150]}
                #im0 = trackerr(im0)
                time_keeper()
                txt1 = f'Operation:           Duration:            '
                txt2 = f'           {machine_status}               {str(status_time)} mins.'
                cv2.putText(im0, txt1, (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255, 255, 255], 2)
                cv2.putText(im0, txt2, (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state[machine_status], 2)
                cv2.polylines(im0, [np.array(area, np.int32)], True, (0, 0, 255), 2)
                vidsave.addframe(im0)
                if "23:59:58"<datetime.datetime.now().strftime("%H:%M:%S")<"23:59:59":
                    dbdata.add_dbdata(["Hazira", "Sheetcutting", f"{camname}", mach_stat[machine_status] , status_time, 0.0])
                    status_time = 0
                cv2.imshow(f'{Camname}_1.4', cv2.resize(im0, dsize=(0,0),fx = 0.5, fy = 0.5))
                if cv2.waitKey(1) & 0xff==ord('q'):
                    dbdata.add_dbdata(["Hazira", "Sheetcutting", f"{camname}", mach_stat[machine_status] , status_time, 0.0])
                    status_time = 0
                    break  # 1 millisecond
            #time.sleep(0.5)
        else:
            #cv2.destroyWindow(Camname)
            print("No Frame...")
            time.sleep(2)
            cap = cv2.VideoCapture(source)


import threading

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbusername', type =str, help='dbusername') 
    parser.add_argument('--dbpassword',type =str,help="dbpasswd")   
    parser.add_argument('--source',type=str,help="source")
    parser.add_argument('--cam', type=str)
    opt = parser.parse_args()

    source1 = r"3.mp4"
    camname = "BSCAM8"

    with torch.no_grad():
        detect(source1, camname)