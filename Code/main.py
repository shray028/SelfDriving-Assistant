import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from moviepy.editor import VideoFileClip
from tkinter import * 
from PIL import Image, ImageTk
from playsound import playsound
# pip install playsound==1.2.2

def merged1():
    global cnt
    cnt = 0
    input_height = 480
    input_width = 640

    # Initialize the parameters

    confThreshold = 0.20  #Confidence threshold
    iouThreshold = 0.50
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image

    red = (0, 0, 255)
    green = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    fontScale = 0.5
    thickness = 1

    triangle = []
    
    # Load names of classes
    classesFile = "traffic_4/obj.names"
    # classesFile = "traffic_4_tiny/obj.names"
    classes_yolo = None
    with open(classesFile, 'rt') as f:
        classes_yolo = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration_yolo = "traffic_4/yolov4-obj.cfg";
    # modelConfiguration_yolo = "traffic_4_tiny/yolov4-tiny-custom.cfg";
    modelWeights_yolo = "traffic_4/yolov4-obj_best.weights";
    # modelWeights_yolo = "traffic_4_tiny/yolov4-tiny-custom_best.weights";
    net_yolo = cv2.dnn.readNetFromDarknet(modelConfiguration_yolo, modelWeights_yolo)
    net_yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net_yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i-1] for i in net.getUnconnectedOutLayers()[0]]
        #print(net.getUnconnectedOutLayers())
        
    def calculate_iou(box_1, box_2):
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou
    
    def get_bounding_box(outs,classes):
        classIds=[]
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    
                    center_x = int(detection[0] * input_width)
                    center_y = int(detection[1] * input_height)
                    width = int(detection[2] * input_width)
                    height = int(detection[3] * input_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    boxes.append([left, top, width, height])


        class_and_box_dict = {}
        for idx in range(len(classIds)):
            box = boxes[idx]
            bottom_left_x = box[0]
            bottom_left_y = box[1]
            width = box[2]
            height = box[3]

            bottom_left = [bottom_left_x,bottom_left_y]
            bottom_right = [bottom_left_x+width,bottom_left_y]

            top_left = [bottom_left_x,bottom_left_y+height]
            top_right = [bottom_left_x+width,bottom_left_y+height]

            class_name = classes[classIds[idx]]
            if class_name in class_and_box_dict.keys():
                class_and_box_dict[class_name].append([top_left,top_right,bottom_right,bottom_left])
            else:
                class_and_box_dict[class_name] = [[top_left,top_right,bottom_right,bottom_left]]
        return class_and_box_dict
    
    def plot_box(frame1,class_and_box_dict):
        
        for key in class_and_box_dict.keys():
            for idx in range(len(class_and_box_dict[key])):
                box = class_and_box_dict[key][idx]
                top_left,top_right,bottom_right,bottom_left = box
                org = (top_left[0],top_left[1]-10)
                color=green
                frame1 = cv2.rectangle(frame1,tuple(bottom_left),tuple(top_right),color,2)
                frame1 = cv2.putText(frame1, key, org, font, 1, red, 2, cv2.LINE_AA) 
            # print(cnt)
            global cnt
            if (cnt%20 == 0):
                audio = f"Audio_files/{key}.mp3"
                playsound(audio,False)
                # print(cnt)
            cnt = cnt + 1

        return frame1
    
    def check_iou_score(class_and_box_dict):
        for key in class_and_box_dict.keys():
            if len(class_and_box_dict[key])>1:
                all_boxes = class_and_box_dict[key]
                pop_idx = []
                final_boxes = []
                for i in range(len(all_boxes)):
                    for j in range(i+1,len(all_boxes)):
                        iou_score = calculate_iou(all_boxes[i],all_boxes[j])
                        if iou_score >= iouThreshold:
                            pop_idx.append(i)
                for m in range(len(all_boxes)):
                    if m not in pop_idx:
                        final_boxes.append(all_boxes[m])

                class_and_box_dict[key] = final_boxes
        return class_and_box_dict
    
    def detect_obj(img):
        frame =cv2.resize(img,(input_width,input_height))
        blob = cv2.dnn.blobFromImage(frame, 1/255, (input_height, input_width), [0,0,0], 1, crop=False)
        net_yolo.setInput(blob)

        outs = net_yolo.forward(getOutputsNames(net_yolo))
        class_and_box_dict = get_bounding_box(outs,classes_yolo)
        class_and_box_dict = check_iou_score(class_and_box_dict)


        frame = plot_box(frame,class_and_box_dict)
        return frame
    
    def line_intersection(line1, line2):
        
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return 0

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
    def calc_distance(pt1,pt2):
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5

    def grey(image):
        #convert to grayscale
        image = np.asarray(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #Apply Gaussian Blur --> Reduce noise and smoothen image
    def gauss(image):
        return cv2.GaussianBlur(image, (3, 3), 0)

    #outline the strongest gradients in the image --> this is where lines in the image are
    def canny(image):
        # edges = cv2.Canny(image,100,150)
        edges = cv2.Canny(image,50,100)
        return edges

    def region(image):
        height, width = image.shape
        #print(height, width)
        #isolate the gradients that correspond to the lane lines
        triangle = np.array([
                           [(100, height), (650, 450), (width, height)]
                           ])

        # triangle = np.array([
        #                     [(300, 650), (715, 450), (980, height)]
        #                     ])
        #create a black image with the same dimensions as original image
        mask = np.zeros_like(image)
        #create a mask (triangle that isolates the region of interest in our image)
        mask = cv2.fillPoly(mask, triangle, 255)
        mask = cv2.bitwise_and(image, mask)
        return mask

    def display_lines(image, lines):
        lines_image = np.zeros_like(image)
        #make sure array isn't empty
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line

                #draw lines on a black image
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    #     print(lines_image)   
    #     imshow('image',lines_image)
        line1 = [lines[0][0],lines[0][1]],[lines[0][2],lines[0][3]]
        line2 = [lines[1][0],lines[1][1]],[lines[1][2],lines[1][3]]
        x,y = line_intersection(line1,line2)
    #     print("anything")
        dist1 = calc_distance([x,y],line1[1])
        dist2 = calc_distance([x,y],line2[1])
        #print(dist1, dist2)
        if dist2 >= dist1*1.25:
            cv2.putText(lines_image,"LeftCurvature", (40,40), font, 1, red, 2, cv2.LINE_AA)
        elif dist1 >= dist2*1.25:
            cv2.putText(lines_image, "RightCurvature", (40,40), font, 1, red, 2, cv2.LINE_AA)
        else :
            cv2.putText(lines_image, "Straight", (40,40), font, 1, red, 2, cv2.LINE_AA)
        cv2.circle(lines_image, (int(x),int(y)), radius=3, color=(0, 0, 255), thickness=-1)
    #     cv2.imshow("Img",lines_image)
    #     print(lines_image)
        return lines_image

    def average(image, lines):
        left = []
        right = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #fit line to points, return slope and y-int
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_int = parameters[1]
            #lines on the right have positive slope, and lines on the left have neg slope
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))
        #takes average among all the columns (column0: slope, column1: y_int)
        right_avg = np.average(right, axis=0)
        left_avg = np.average(left, axis=0)
        
        #create lines based on averages calculates
        left_line = make_points(image, left_avg)
        right_line = make_points(image, right_avg)
        return np.array([left_line, right_line])

    def make_points(image, average):
        slope, y_int = average
        #print(slope)
        y1 = image.shape[0]
        #how long we want our lines to be --> 3/5 the size of the image
        y2 = int(y1 * (3/5))
        #determine algebraically
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)
        return np.array([x1, y1, x2, y2])

    def plot_lanes(frame):
        frame = cv2.resize(frame,(1280,720))
        try:
            grey_img = grey(frame)
            gaus = gauss(grey_img)
            edges = canny(gaus)
            # cv2.imshow("image",edges)
            isolated = region(edges)
            #region of interest, bin size (P, theta), min intersections needed, placeholder array, 
            lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=5)
            averaged_lines = average(frame, lines)
            black_lines = display_lines(frame, averaged_lines)
            #taking wighted sum of original image and lane lines image
            lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
        except Exception as e:
    #         print(e)
            lanes = cv2.putText(frame, "LaneChange", (20,40), font, 1, red, 2, cv2.LINE_AA)
        return lanes
    
    def main(frame):
        img = detect_obj(frame)
        frame = plot_lanes(img)
        return frame
    
    camera = cv2.VideoCapture("input\IndianHighway.mp4")

    ret, frame = camera.read()

    # check of live video
    while(camera.isOpened()):
    # Capture frame-by-frame
        ret, frame = camera.read()
        if ret == True:
            # Display the resulting frame
            frame = main(frame)
            #frame1 = cv2.resize(frame, (640, 480))
            cv2.imshow('Frame',frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break

    camera.release()
    cv2.destroyAllWindows()
    
    
    # # check for recorded video
    # video_output = 'output/rach.mp4'
    # clip1 = VideoFileClip("Test1.mp4")

    # clip = clip1.fl_image(main) #NOTE: it should be in BGR format
    # clip.write_videofile(video_output, audio=True)

def merged2():
    global cnt
    cnt = 0
    input_height = 480
    input_width = 640

    # Initialize the parameters

    confThreshold = 0.20  #Confidence threshold
    iouThreshold = 0.20
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image

    red = (0, 0, 255)
    green = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    fontScale = 0.5
    thickness = 1

    triangle = []
    
    # Load names of classes
    classesFile = "traffic_4/obj.names"
    # classesFile = "traffic_4_tiny/obj.names"
    classes_yolo = None
    with open(classesFile, 'rt') as f:
        classes_yolo = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration_yolo = "traffic_4/yolov4-obj.cfg";
    # modelConfiguration_yolo = "traffic_4_tiny/yolov4-tiny-custom.cfg";
    modelWeights_yolo = "traffic_4/yolov4-obj_best.weights";
    # modelWeights_yolo = "traffic_4_tiny/yolov4-tiny-custom_best.weights";
    net_yolo = cv2.dnn.readNetFromDarknet(modelConfiguration_yolo, modelWeights_yolo)
    net_yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net_yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i-1] for i in net.getUnconnectedOutLayers()[0]]
        #print(net.getUnconnectedOutLayers())
        
    def calculate_iou(box_1, box_2):
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou
    
    def get_bounding_box(outs,classes):
        classIds=[]
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    
                    center_x = int(detection[0] * input_width)
                    center_y = int(detection[1] * input_height)
                    width = int(detection[2] * input_width)
                    height = int(detection[3] * input_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    boxes.append([left, top, width, height])


        class_and_box_dict = {}
        for idx in range(len(classIds)):
            box = boxes[idx]
            bottom_left_x = box[0]
            bottom_left_y = box[1]
            width = box[2]
            height = box[3]

            bottom_left = [bottom_left_x,bottom_left_y]
            bottom_right = [bottom_left_x+width,bottom_left_y]

            top_left = [bottom_left_x,bottom_left_y+height]
            top_right = [bottom_left_x+width,bottom_left_y+height]

            class_name = classes[classIds[idx]]
            if class_name in class_and_box_dict.keys():
                class_and_box_dict[class_name].append([top_left,top_right,bottom_right,bottom_left])
            else:
                class_and_box_dict[class_name] = [[top_left,top_right,bottom_right,bottom_left]]
        return class_and_box_dict
    
    def plot_box(frame1,class_and_box_dict):
        
        for key in class_and_box_dict.keys():
            for idx in range(len(class_and_box_dict[key])):
                box = class_and_box_dict[key][idx]
                top_left,top_right,bottom_right,bottom_left = box
                org = (top_left[0],top_left[1]-10)
                color=green
                frame1 = cv2.rectangle(frame1,tuple(bottom_left),tuple(top_right),color,2)
                frame1 = cv2.putText(frame1, key, org, font, 1, red, 2, cv2.LINE_AA) 
            # print(cnt)
            global cnt
            if (cnt%20 == 0):
                audio = f"Audio_files/{key}.mp3"
                playsound(audio,False)
                # print(cnt)
            cnt = cnt + 1

        return frame1
    
    def check_iou_score(class_and_box_dict):
        for key in class_and_box_dict.keys():
            if len(class_and_box_dict[key])>1:
                all_boxes = class_and_box_dict[key]
                pop_idx = []
                final_boxes = []
                for i in range(len(all_boxes)):
                    for j in range(i+1,len(all_boxes)):
                        iou_score = calculate_iou(all_boxes[i],all_boxes[j])
                        if iou_score >= iouThreshold:
                            pop_idx.append(i)
                for m in range(len(all_boxes)):
                    if m not in pop_idx:
                        final_boxes.append(all_boxes[m])

                class_and_box_dict[key] = final_boxes
        return class_and_box_dict
    
    def detect_obj(img):
        frame =cv2.resize(img,(input_width,input_height))
        blob = cv2.dnn.blobFromImage(frame, 1/255, (input_height, input_width), [0,0,0], 1, crop=False)
        net_yolo.setInput(blob)

        outs = net_yolo.forward(getOutputsNames(net_yolo))
        class_and_box_dict = get_bounding_box(outs,classes_yolo)
        class_and_box_dict = check_iou_score(class_and_box_dict)


        frame = plot_box(frame,class_and_box_dict)
        return frame
    
    def line_intersection(line1, line2):
        
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return 0

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
    def calc_distance(pt1,pt2):
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5

    def grey(image):
        #convert to grayscale
        image = np.asarray(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #Apply Gaussian Blur --> Reduce noise and smoothen image
    def gauss(image):
        return cv2.GaussianBlur(image, (3, 3), 0)

    #outline the strongest gradients in the image --> this is where lines in the image are
    def canny(image):
        # edges = cv2.Canny(image,100,150)
        edges = cv2.Canny(image,50,100)
        return edges

    def region(image):
        height, width = image.shape
        #print(height, width)
        #isolate the gradients that correspond to the lane lines
        # triangle = np.array([
        #                    [(100, height), (650, 450), (width, height)]
        #                    ])

        triangle = np.array([
                            [(300, 650), (715, 450), (980, height)]
                            ])
        #create a black image with the same dimensions as original image
        mask = np.zeros_like(image)
        #create a mask (triangle that isolates the region of interest in our image)
        mask = cv2.fillPoly(mask, triangle, 255)
        mask = cv2.bitwise_and(image, mask)
        return mask

    def display_lines(image, lines):
        lines_image = np.zeros_like(image)
        #make sure array isn't empty
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line

                #draw lines on a black image
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    #     print(lines_image)   
    #     imshow('image',lines_image)
        line1 = [lines[0][0],lines[0][1]],[lines[0][2],lines[0][3]]
        line2 = [lines[1][0],lines[1][1]],[lines[1][2],lines[1][3]]
        x,y = line_intersection(line1,line2)
    #     print("anything")
        dist1 = calc_distance([x,y],line1[1])
        dist2 = calc_distance([x,y],line2[1])
        #print(dist1, dist2)
        if dist2 >= dist1*1.25:
            cv2.putText(lines_image,"LeftCurvature", (40,40), font, 1, red, 2, cv2.LINE_AA)
        elif dist1 >= dist2*1.25:
            cv2.putText(lines_image, "RightCurvature", (40,40), font, 1, red, 2, cv2.LINE_AA)
        else :
            cv2.putText(lines_image, "Straight", (40,40), font, 1, red, 2, cv2.LINE_AA)
        cv2.circle(lines_image, (int(x),int(y)), radius=3, color=(0, 0, 255), thickness=-1)
    #     cv2.imshow("Img",lines_image)
    #     print(lines_image)
        return lines_image

    def average(image, lines):
        left = []
        right = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #fit line to points, return slope and y-int
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_int = parameters[1]
            #lines on the right have positive slope, and lines on the left have neg slope
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))
        #takes average among all the columns (column0: slope, column1: y_int)
        right_avg = np.average(right, axis=0)
        left_avg = np.average(left, axis=0)
        
        #create lines based on averages calculates
        left_line = make_points(image, left_avg)
        right_line = make_points(image, right_avg)
        return np.array([left_line, right_line])

    def make_points(image, average):
        slope, y_int = average
        #print(slope)
        y1 = image.shape[0]
        #how long we want our lines to be --> 3/5 the size of the image
        y2 = int(y1 * (3/5))
        #determine algebraically
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)
        return np.array([x1, y1, x2, y2])

    def plot_lanes(frame):
        frame = cv2.resize(frame,(1280,720))
        try:
            grey_img = grey(frame)
            gaus = gauss(grey_img)
            edges = canny(gaus)
            # cv2.imshow("image",edges)
            isolated = region(edges)
            #region of interest, bin size (P, theta), min intersections needed, placeholder array, 
            lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=5)
            averaged_lines = average(frame, lines)
            black_lines = display_lines(frame, averaged_lines)
            #taking wighted sum of original image and lane lines image
            lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
        except Exception as e:
    #         print(e)
            lanes = cv2.putText(frame, "LaneChange", (20,40), font, 1, red, 2, cv2.LINE_AA)
        return lanes
    
    def main(frame):
        img = detect_obj(frame)
        frame = plot_lanes(img)
        return frame

    camera = cv2.VideoCapture("input\Test7.mp4")

    ret, frame = camera.read()

    # check of live video
    while(camera.isOpened()):
    # Capture frame-by-frame
        ret, frame = camera.read()
        if ret == True:
            # Display the resulting frame
            frame = main(frame)
            #frame1 = cv2.resize(frame, (640, 480))
            cv2.imshow('Frame',frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break

    camera.release()
    cv2.destroyAllWindows()
    
    
    # # check for recorded video
    # video_output = 'output/rach.mp4'
    # clip1 = VideoFileClip("Test1.mp4")

    # clip = clip1.fl_image(main) #NOTE: it should be in BGR format
    # clip.write_videofile(video_output, audio=True)

def merged3():
    global cnt
    cnt = 0
    input_height = 480
    input_width = 640

    # Initialize the parameters

    confThreshold = 0.20  #Confidence threshold
    iouThreshold = 0.20
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image

    red = (0, 0, 255)
    green = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    fontScale = 0.5
    thickness = 1

    triangle = []
    
    # Load names of classes
    # classesFile = "traffic_4/obj.names"
    classesFile = "traffic_4_tiny/obj.names"
    classes_yolo = None
    with open(classesFile, 'rt') as f:
        classes_yolo = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    # modelConfiguration_yolo = "traffic_4/yolov4-obj.cfg";
    modelConfiguration_yolo = "traffic_4_tiny/yolov4-tiny-custom.cfg";
    # modelWeights_yolo = "traffic_4/yolov4-obj_best.weights";
    modelWeights_yolo = "traffic_4_tiny/yolov4-tiny-custom_best.weights";
    net_yolo = cv2.dnn.readNetFromDarknet(modelConfiguration_yolo, modelWeights_yolo)
    net_yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net_yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i-1] for i in net.getUnconnectedOutLayers()[0]]
        #print(net.getUnconnectedOutLayers())
        
    def calculate_iou(box_1, box_2):
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou
    
    def get_bounding_box(outs,classes):
        classIds=[]
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    
                    center_x = int(detection[0] * input_width)
                    center_y = int(detection[1] * input_height)
                    width = int(detection[2] * input_width)
                    height = int(detection[3] * input_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    boxes.append([left, top, width, height])


        class_and_box_dict = {}
        for idx in range(len(classIds)):
            box = boxes[idx]
            bottom_left_x = box[0]
            bottom_left_y = box[1]
            width = box[2]
            height = box[3]

            bottom_left = [bottom_left_x,bottom_left_y]
            bottom_right = [bottom_left_x+width,bottom_left_y]

            top_left = [bottom_left_x,bottom_left_y+height]
            top_right = [bottom_left_x+width,bottom_left_y+height]

            class_name = classes[classIds[idx]]
            if class_name in class_and_box_dict.keys():
                class_and_box_dict[class_name].append([top_left,top_right,bottom_right,bottom_left])
            else:
                class_and_box_dict[class_name] = [[top_left,top_right,bottom_right,bottom_left]]
        return class_and_box_dict
    
    def plot_box(frame1,class_and_box_dict):
        
        for key in class_and_box_dict.keys():
            for idx in range(len(class_and_box_dict[key])):
                box = class_and_box_dict[key][idx]
                top_left,top_right,bottom_right,bottom_left = box
                org = (top_left[0],top_left[1]-10)
                color=green
                frame1 = cv2.rectangle(frame1,tuple(bottom_left),tuple(top_right),color,2)
                frame1 = cv2.putText(frame1, key, org, font, 1, red, 2, cv2.LINE_AA) 
            # print(cnt)
            global cnt
            if (cnt%20 == 0):
                audio = f"Audio_files/{key}.mp3"
                playsound(audio,False)
                # print(cnt)
            cnt = cnt + 1

        return frame1
    
    def check_iou_score(class_and_box_dict):
        for key in class_and_box_dict.keys():
            if len(class_and_box_dict[key])>1:
                all_boxes = class_and_box_dict[key]
                pop_idx = []
                final_boxes = []
                for i in range(len(all_boxes)):
                    for j in range(i+1,len(all_boxes)):
                        iou_score = calculate_iou(all_boxes[i],all_boxes[j])
                        if iou_score >= iouThreshold:
                            pop_idx.append(i)
                for m in range(len(all_boxes)):
                    if m not in pop_idx:
                        final_boxes.append(all_boxes[m])

                class_and_box_dict[key] = final_boxes
        return class_and_box_dict
    
    def detect_obj(img):
        frame =cv2.resize(img,(input_width,input_height))
        blob = cv2.dnn.blobFromImage(frame, 1/255, (input_height, input_width), [0,0,0], 1, crop=False)
        net_yolo.setInput(blob)

        outs = net_yolo.forward(getOutputsNames(net_yolo))
        class_and_box_dict = get_bounding_box(outs,classes_yolo)
        class_and_box_dict = check_iou_score(class_and_box_dict)


        frame = plot_box(frame,class_and_box_dict)
        return frame
    
    def line_intersection(line1, line2):
        
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return 0

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
    def calc_distance(pt1,pt2):
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5

    def grey(image):
        #convert to grayscale
        image = np.asarray(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #Apply Gaussian Blur --> Reduce noise and smoothen image
    def gauss(image):
        return cv2.GaussianBlur(image, (3, 3), 0)

    #outline the strongest gradients in the image --> this is where lines in the image are
    def canny(image):
        # edges = cv2.Canny(image,100,150)
        edges = cv2.Canny(image,50,100)
        return edges

    def region(image):
        height, width = image.shape
        #print(height, width)
        #isolate the gradients that correspond to the lane lines
        # triangle = np.array([
        #                    [(100, height), (650, 450), (width, height)]
        #                    ])

        triangle = np.array([
                            [(300, 650), (715, 450), (980, height)]
                            ])
        #create a black image with the same dimensions as original image
        mask = np.zeros_like(image)
        #create a mask (triangle that isolates the region of interest in our image)
        mask = cv2.fillPoly(mask, triangle, 255)
        mask = cv2.bitwise_and(image, mask)
        return mask

    def display_lines(image, lines):
        lines_image = np.zeros_like(image)
        #make sure array isn't empty
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line

                #draw lines on a black image
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    #     print(lines_image)   
    #     imshow('image',lines_image)
        line1 = [lines[0][0],lines[0][1]],[lines[0][2],lines[0][3]]
        line2 = [lines[1][0],lines[1][1]],[lines[1][2],lines[1][3]]
        x,y = line_intersection(line1,line2)
    #     print("anything")
        dist1 = calc_distance([x,y],line1[1])
        dist2 = calc_distance([x,y],line2[1])
        #print(dist1, dist2)
        if dist2 >= dist1*1.25:
            cv2.putText(lines_image,"LeftCurvature", (40,40), font, 1, red, 2, cv2.LINE_AA)
        elif dist1 >= dist2*1.25:
            cv2.putText(lines_image, "RightCurvature", (40,40), font, 1, red, 2, cv2.LINE_AA)
        else :
            cv2.putText(lines_image, "Straight", (40,40), font, 1, red, 2, cv2.LINE_AA)
        cv2.circle(lines_image, (int(x),int(y)), radius=3, color=(0, 0, 255), thickness=-1)
    #     cv2.imshow("Img",lines_image)
    #     print(lines_image)
        return lines_image

    def average(image, lines):
        left = []
        right = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #fit line to points, return slope and y-int
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_int = parameters[1]
            #lines on the right have positive slope, and lines on the left have neg slope
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))
        #takes average among all the columns (column0: slope, column1: y_int)
        right_avg = np.average(right, axis=0)
        left_avg = np.average(left, axis=0)
        
        #create lines based on averages calculates
        left_line = make_points(image, left_avg)
        right_line = make_points(image, right_avg)
        return np.array([left_line, right_line])

    def make_points(image, average):
        slope, y_int = average
        #print(slope)
        y1 = image.shape[0]
        #how long we want our lines to be --> 3/5 the size of the image
        y2 = int(y1 * (3/5))
        #determine algebraically
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)
        return np.array([x1, y1, x2, y2])

    def plot_lanes(frame):
        frame = cv2.resize(frame,(1280,720))
        try:
            grey_img = grey(frame)
            gaus = gauss(grey_img)
            edges = canny(gaus)
            # cv2.imshow("image",edges)
            isolated = region(edges)
            #region of interest, bin size (P, theta), min intersections needed, placeholder array, 
            lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=5)
            averaged_lines = average(frame, lines)
            black_lines = display_lines(frame, averaged_lines)
            #taking wighted sum of original image and lane lines image
            lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
        except Exception as e:
    #         print(e)
            lanes = cv2.putText(frame, "LaneChange", (20,40), font, 1, red, 2, cv2.LINE_AA)
        return lanes
    
    def main(frame):
        img = detect_obj(frame)
        img = cv2.resize(img,(1280,720))
        # frame = plot_lanes(img)
        return img

    camera = cv2.VideoCapture("input\zebra1.mp4")
    ret, frame = camera.read()

    # check of live video
    while(camera.isOpened()):
    # Capture frame-by-frame
        ret, frame = camera.read()
        if ret == True:
            # Display the resulting frame
            frame = main(frame)
            #frame1 = cv2.resize(frame, (640, 480))
            cv2.imshow('Frame',frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break

    camera.release()
    cv2.destroyAllWindows()
    
    
    # # check for recorded video
    # video_output = 'output/rach.mp4'
    # clip1 = VideoFileClip("Test1.mp4")

    # clip = clip1.fl_image(main) #NOTE: it should be in BGR format
    # clip.write_videofile(video_output, audio=True)
    
def ObjectDetection():
    global cnt
    cnt = 0
    input_height = 480
    input_width = 640

    # Initialize the parameters

    confThreshold = 0.20  #Confidence threshold
    iouThreshold = 0.20
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image

    red = (0, 0, 255)
    green = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    fontScale = 0.5
    thickness = 1

    triangle = []
    
    # Load names of classes
    # classesFile = "traffic_4/obj.names"
    classesFile = "traffic_4_tiny/obj.names"
    classes_yolo = None
    with open(classesFile, 'rt') as f:
        classes_yolo = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    # modelConfiguration_yolo = "traffic_4/yolov4-obj.cfg";
    modelConfiguration_yolo = "traffic_4_tiny/yolov4-tiny-custom.cfg";
    # modelWeights_yolo = "traffic_4/yolov4-obj_best2.weights";
    modelWeights_yolo = "traffic_4_tiny/yolov4-tiny-custom_best.weights";
    net_yolo = cv2.dnn.readNetFromDarknet(modelConfiguration_yolo, modelWeights_yolo)
    net_yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net_yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i-1] for i in net.getUnconnectedOutLayers()[0]]
        #print(net.getUnconnectedOutLayers())
        
    def calculate_iou(box_1, box_2):
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou
    
    def get_bounding_box(outs,classes):
        classIds=[]
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    
                    center_x = int(detection[0] * input_width)
                    center_y = int(detection[1] * input_height)
                    width = int(detection[2] * input_width)
                    height = int(detection[3] * input_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    boxes.append([left, top, width, height])


        class_and_box_dict = {}
        for idx in range(len(classIds)):
            box = boxes[idx]
            bottom_left_x = box[0]
            bottom_left_y = box[1]
            width = box[2]
            height = box[3]

            bottom_left = [bottom_left_x,bottom_left_y]
            bottom_right = [bottom_left_x+width,bottom_left_y]

            top_left = [bottom_left_x,bottom_left_y+height]
            top_right = [bottom_left_x+width,bottom_left_y+height]

            class_name = classes[classIds[idx]]
            if class_name in class_and_box_dict.keys():
                class_and_box_dict[class_name].append([top_left,top_right,bottom_right,bottom_left])
            else:
                class_and_box_dict[class_name] = [[top_left,top_right,bottom_right,bottom_left]]
        return class_and_box_dict
    
    def plot_box(frame1,class_and_box_dict):
        
        for key in class_and_box_dict.keys():
            for idx in range(len(class_and_box_dict[key])):
                box = class_and_box_dict[key][idx]
                top_left,top_right,bottom_right,bottom_left = box
                org = (top_left[0],top_left[1]-10)
                color=green
                frame1 = cv2.rectangle(frame1,tuple(bottom_left),tuple(top_right),color,2)
                frame1 = cv2.putText(frame1, key, org, font, 1, red, 2, cv2.LINE_AA) 
            # print(cnt)
            global cnt
            if (cnt%20 == 0):
                audio = f"Audio_files/{key}.mp3"
                playsound(audio,False)
                # print(cnt)
            cnt = cnt + 1

        return frame1
    
    def check_iou_score(class_and_box_dict):
        for key in class_and_box_dict.keys():
            if len(class_and_box_dict[key])>1:
                all_boxes = class_and_box_dict[key]
                pop_idx = []
                final_boxes = []
                for i in range(len(all_boxes)):
                    for j in range(i+1,len(all_boxes)):
                        iou_score = calculate_iou(all_boxes[i],all_boxes[j])
                        if iou_score >= iouThreshold:
                            pop_idx.append(i)
                for m in range(len(all_boxes)):
                    if m not in pop_idx:
                        final_boxes.append(all_boxes[m])

                class_and_box_dict[key] = final_boxes
        return class_and_box_dict
    
    def detect_obj(img):
        frame =cv2.resize(img,(input_width,input_height))
        blob = cv2.dnn.blobFromImage(frame, 1/255, (input_height, input_width), [0,0,0], 1, crop=False)
        net_yolo.setInput(blob)

        outs = net_yolo.forward(getOutputsNames(net_yolo))
        class_and_box_dict = get_bounding_box(outs,classes_yolo)
        class_and_box_dict = check_iou_score(class_and_box_dict)


        frame = plot_box(frame,class_and_box_dict)
        return frame
    
    def line_intersection(line1, line2):
        
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return 0

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
    def calc_distance(pt1,pt2):
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5

    def grey(image):
        #convert to grayscale
        image = np.asarray(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #Apply Gaussian Blur --> Reduce noise and smoothen image
    def gauss(image):
        return cv2.GaussianBlur(image, (3, 3), 0)

    #outline the strongest gradients in the image --> this is where lines in the image are
    def canny(image):
        # edges = cv2.Canny(image,100,150)
        edges = cv2.Canny(image,50,100)
        return edges

    def region(image):
        height, width = image.shape
        #print(height, width)
        #isolate the gradients that correspond to the lane lines
        # triangle = np.array([
        #                    [(100, height), (650, 450), (width, height)]
        #                    ])

        triangle = np.array([
                            [(300, 650), (715, 450), (980, height)]
                            ])
        #create a black image with the same dimensions as original image
        mask = np.zeros_like(image)
        #create a mask (triangle that isolates the region of interest in our image)
        mask = cv2.fillPoly(mask, triangle, 255)
        mask = cv2.bitwise_and(image, mask)
        return mask

    def display_lines(image, lines):
        lines_image = np.zeros_like(image)
        #make sure array isn't empty
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line

                #draw lines on a black image
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    #     print(lines_image)   
    #     imshow('image',lines_image)
        line1 = [lines[0][0],lines[0][1]],[lines[0][2],lines[0][3]]
        line2 = [lines[1][0],lines[1][1]],[lines[1][2],lines[1][3]]
        x,y = line_intersection(line1,line2)
    #     print("anything")
        dist1 = calc_distance([x,y],line1[1])
        dist2 = calc_distance([x,y],line2[1])
        #print(dist1, dist2)
        if dist2 >= dist1*1.25:
            cv2.putText(lines_image,"LeftCurvature", (40,40), font, 1, red, 2, cv2.LINE_AA)
        elif dist1 >= dist2*1.25:
            cv2.putText(lines_image, "RightCurvature", (40,40), font, 1, red, 2, cv2.LINE_AA)
        else :
            cv2.putText(lines_image, "Straight", (40,40), font, 1, red, 2, cv2.LINE_AA)
        cv2.circle(lines_image, (int(x),int(y)), radius=3, color=(0, 0, 255), thickness=-1)
    #     cv2.imshow("Img",lines_image)
    #     print(lines_image)
        return lines_image

    def average(image, lines):
        left = []
        right = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #fit line to points, return slope and y-int
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_int = parameters[1]
            #lines on the right have positive slope, and lines on the left have neg slope
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))
        #takes average among all the columns (column0: slope, column1: y_int)
        right_avg = np.average(right, axis=0)
        left_avg = np.average(left, axis=0)
        
        #create lines based on averages calculates
        left_line = make_points(image, left_avg)
        right_line = make_points(image, right_avg)
        return np.array([left_line, right_line])

    def make_points(image, average):
        slope, y_int = average
        #print(slope)
        y1 = image.shape[0]
        #how long we want our lines to be --> 3/5 the size of the image
        y2 = int(y1 * (3/5))
        #determine algebraically
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)
        return np.array([x1, y1, x2, y2])

    def plot_lanes(frame):
        frame = cv2.resize(frame,(1280,720))
        try:
            grey_img = grey(frame)
            gaus = gauss(grey_img)
            edges = canny(gaus)
            # cv2.imshow("image",edges)
            isolated = region(edges)
            #region of interest, bin size (P, theta), min intersections needed, placeholder array, 
            lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=5)
            averaged_lines = average(frame, lines)
            black_lines = display_lines(frame, averaged_lines)
            #taking wighted sum of original image and lane lines image
            lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
        except Exception as e:
    #         print(e)
            lanes = cv2.putText(frame, "LaneChange", (20,40), font, 1, red, 2, cv2.LINE_AA)
        return lanes
    
    def main(frame):
        img = detect_obj(frame)
        # frame = plot_lanes(img)
        return img
    
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()

    # check of live video
    while(camera.isOpened()):
    # Capture frame-by-frame
        ret, frame = camera.read()
        if ret == True:
            # Display the resulting frame
            frame = main(frame)
            #frame1 = cv2.resize(frame, (640, 480))
            cv2.imshow('Frame',frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break

    camera.release()
    cv2.destroyAllWindows()
    
    
    # # check for recorded video
    # video_output = 'output/rach.mp4'
    # clip1 = VideoFileClip("Test1.mp4")

    # clip = clip1.fl_image(main) #NOTE: it should be in BGR format
    # clip.write_videofile(video_output, audio=True)

root = Tk()
root.title('Self Driving Vehicle Assistant System')
root.iconbitmap('car.ico')            
 
# Open window having dimension 100x100
root.geometry('1000x500')

image = Image.open("sit.jpg")
photo = ImageTk.PhotoImage(image)

label = Label(image=photo)
label.pack()
Label(root,fg='red',text = "SIDDAGANGA INSTITUTE OF TECHNOLOGY" ,font = ('Helvetica', 20), width=40, anchor = 'c').pack()
Label(root,text = " " ,font = ('Helvetica', 15), width=40, anchor = 'c').pack()
Label(root,text = "Self Driving Vehicle Assistant System" ,font = ('Helvetica', 15), width=40, anchor = 'c').pack()

 
frame1=LabelFrame(root,width=500,height=500, text = "Video")
frame1.pack(fill = 'x', expand = "yes", padx = 20)
# Create a Button
btn1 = Button(frame1, text = 'Sample 1', bd = '5',
                          command = merged1)
# Set the position of button on the top of window.  
btn1.grid(row = 0, column=0, padx = 10, pady = 10)

btn2 = Button(frame1, text = 'Sample 2', bd = '5',
                          command = merged2)
# Set the position of button on the top of window.  
btn2.grid(row = 0, column=1, padx = 10, pady = 10)

btn3 = Button(frame1, text = 'Sample 3', bd = '5',
                          command = merged3)
# Set the position of button on the top of window.  
btn3.grid(row = 0, column=2, padx = 10, pady = 10)

frame2=LabelFrame(root,width=500,height=500, text = "Webcam")
frame2.pack(fill = 'x', expand = "yes", padx = 20)   

# Create a Button
btn = Button(frame2, text = 'Run on a webcam', bd = '5',
                          command = ObjectDetection)
# Set the position of button on the top of window.  
btn.grid(row = 0, column=0, padx = 10, pady = 10)

root.mainloop()