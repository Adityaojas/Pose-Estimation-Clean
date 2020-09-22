#import the libraries
from scipy import signal
import pandas as pd
import numpy as np
import progressbar
import cv2, os
import scipy
import csv

circle_color, line_color = (0,255,255), (0,255,0)
window_length, polyorder = 11, 2

video_path = 'samples/ex_11.mov'
csv_path = 'output/out_11.csv'
df = pd.read_csv(csv_path)

# Apply the filter on the data frame to make it smooth
for i in range(30):
    df[str(i)] = signal.savgol_filter(df[str(i)], window_length, polyorder)

cleaned_points = []

for i in range(df.shape[0]):
    row = np.array(df.values[i], int)
    points = list(zip(row[:15], row[15:]))
    
    if points[2][0] > points[5][0]:
        
        #temp = points[2]
        #points[2] = points[5]
        #points[5] = temp
        
        temp = points[3]
        points[3] = points[6]
        points[6] = temp
        
        temp = points[4]
        points[4] = points[7]
        points[7] = temp
        
        temp = points[2]
        points[2] = points[5]
        points[5] = temp
        
        temp = points[8]
        points[8] = points[11]
        points[11] = temp
        
        temp = points[9]
        points[9] = points[12]
        points[12] = temp
        
        temp = points[10]
        points[10] = points[13]
        points[13] = temp
    
    cleaned_points.append(points)

cap = cv2.VideoCapture(video_path)

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

ok, frame = cap.read()
(frameHeight, frameWidth) = frame.shape[:2]
h = 500
w = int((h/frameHeight) * frameWidth)

# Define the output
out_path = 'output/out_11_cleaned.mp4'
output = cv2.VideoWriter(out_path, 0, fps, (w, h))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = None
(f_h, f_w) = (h, w)
zeros = None

# There are 15 points in the skeleton
pairs = [[0,1], # head
         [1,2],[1,5], # sholders
         [2,3],[3,4],[5,6],[6,7], # arms
         [1,14],[14,11],[14,8], # hips
         [8,9],[9,10],[11,12],[12,13]] # legs


frame_number = 0

while True:
    ok, frame = cap.read()
    
    if not ok:
        break
    
    frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)
    frame_copy = np.copy(frame)

    points = cleaned_points[frame_number]

    for i in range(len(points)):
        xy = tuple(np.array([points[i][0], points[i][1]], int))
        cv2.circle(frame_copy, xy, 2, circle_color, -1)

    for pair in pairs:
        partA = pair[0]
        partB = pair[1]
        cv2.line(frame_copy, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)

       
    if writer is None:
        writer = cv2.VideoWriter(out_path, fourcc, fps,
                                 (f_w, f_h), True)
        zeros = np.zeros((f_h, f_w), dtype="uint8")
    
    writer.write(cv2.resize(frame_copy,(f_w, f_h)))
    
    cv2.imshow('Output-Skeleton', frame_copy)
    
    k = cv2.waitKey(100)
    if k == 27:
        break
    
    frame_number+=1

cv2.destroyAllWindows()
