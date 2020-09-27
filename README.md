# Pose-Estimation-Clean
Pose Estimation with cleaner outputs using Savgol filter

Here's the Medium article for reference, and understanding the code: https://medium.com/@adityaojassharma/cleaner-pose-estimation-using-open-pose-6d239cc33fe6

The first step would be to download the MPI .caffemodel wights from http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel. Move the file to the models directory.

You're good to go, run the open_pose.py file to get a raw output video and a .csv file consisting of the coordinates from each frame. You'd notice the output visualisation is not clean. That's normal because OpenPose sometimes messes up and confuses between different joints, and that make the output kinda disjoint.

Don't worry, the other file, clean_ouput.py comes in here. It'll take the output csv file from open_pose.py, apply the Savgol filter to smoothen the subsequent coordinates, and output an extremely, aesthetically composed video.

To understand the whole code, please refer to my Medium article linked above!

# Here's the youtube video for a simple application I made using Pose Estimation to keep track of my Jump Rope Progress
https://youtu.be/gEhNntwJSdY
