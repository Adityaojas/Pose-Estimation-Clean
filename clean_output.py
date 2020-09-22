#import the libraries
from scipy.spatial import distance as dist
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

"""
jump = [0]*(df.shape[0])
thresh = 0.8
"""

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
 
"""
    if 0.0 in row:
        continue
    
    if points[2][0]<points[4][0] or points[5][0]>points[7][0]:
        continue
    
    ref = points[1]
    ref_2 = points[14]
    ref_3 = ((points[8][0]+points[11][0])//2, (points[8][1]+points[11][1])//2) 
    
    lh = points[7]
    rh = points[4]
    
    a = int(dist.euclidean(lh, rh))
    
    b1 = int(dist.euclidean(ref, rh))
    c1 = int(dist.euclidean(ref, lh))
    s1 = (a+b1+c1) / 2
    sq_area_1 = s1*(s1-a)*(s1-b1)*(s1-c1)
    if sq_area_1 > 0:
        ar_1 = int((sq_area_1) ** 0.5)
    else:
        ar_1 = 0
    
    b2 = int(dist.euclidean(ref_2, rh))
    c2 = int(dist.euclidean(ref_2, lh))
    s2 = (a+b2+c2) / 2
    sq_area_2 = s2*(s2-a)*(s2-b2)*(s2-c2)
    if sq_area_2 > 0:
        ar_2 = int((sq_area_2) ** 0.5)
    else:
        ar_2 = 1
    
    b3 = int(dist.euclidean(ref_3, rh))
    c3 = int(dist.euclidean(ref_3, lh))
    s3 = (a+b3+c3) / 2
    sq_area_3 = s3*(s3-a)*(s3-b3)*(s3-c3)
    if sq_area_3 > 0:
        ar_3 = int((sq_area_3) ** 0.5)
    else:
        ar_3 = 1
        
    ratio = ar_3/ar_2
    
    if ratio<thresh:
        jump[i] = 1
    else:
        continue
"""

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

"""
jumps = 0
j=0


l_swing = False
r_swing = False
cross = False

lss_count = 0
rss_count = 0
ch_count = 0

def mess(points):
    if points[0][1]>points[1][1] or points[1][1]>points[14][1]:
        return True
    if points[0][1]>points[2][1] or points[2][1]>points[3][1]:
        return True
    if points[0][1]>points[5][1] or points[5][1]>points[6][1]:
        return True
    if points[14][1]>points[8][1] or points[8][1]>points[9][1] or points[9][1]>points[10][1]:
        return True
    if points[14][1]>points[11][1] or points[11][1]>points[12][1] or points[12][1]>points[13][1]:
        return True
    else:
        return False    
"""

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

    """       
    swing_ref = points[14]
    swing_ref_x = swing_ref[0]
    swing_ref_y = swing_ref[1]
    
    left_should = points[5]
    left_should_x = left_should[0]
    left_should_y = left_should[1]
    
    right_should = points[2]
    right_should_x = left_should[0]
    right_should_y = left_should[1]
    
    left_hand = points[7]
    left_hand_x = left_hand[0]
    left_hand_y = left_hand[1]
    
    right_hand = points[4]
    right_hand_x = right_hand[0]
    right_hand_y = right_hand[1]
    
    if not mess(points):
        
        if left_hand_x > swing_ref_x and right_hand_x < swing_ref_x:
            l_swing = False
            r_swing = False
            cross = False
        
        if l_swing == False:
            if left_hand_x < swing_ref_x and right_hand_x < right_should_x:
                rss_count += 1
                l_swing = True
        
        if r_swing == False:
            if right_hand_x > swing_ref_x and left_hand_x > left_should_x:
                lss_count += 1
                r_swing = True
                
        if cross == False:
            if right_hand_x > swing_ref_x and left_hand_x < swing_ref_x:
                ch_count += 1
                cross = True
        
        if j == 0:
            if jump[frame_number] ==1:
                jumps += 1
    
    j = jump[frame_number]
    
    cv2.putText(frame_copy,'Jumps: {}'.format(jumps), 
                (w//2+20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame_copy,'Left Swings: {}'.format(lss_count), 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame_copy,'Right Swings: {}'.format(rss_count), 
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame_copy,'Cross Hands: {}'.format(ch_count), 
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    """    
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
