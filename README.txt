### Terminal 1

    roscore

### Terminal 2
##### Open calib_l2c.launch and enter K (flattened camera matrix array) obtained via intrinsic calibration 
##### DO NOT USE camera_matrix from ost.yaml file as K since it doesn't include distortion parameters and hence running projection in next step on rect (instead of raw) image will not be correct

    roslaunch calib_lidar_cam calib_l2c.launch

### Terminal 3
##### Open calib_l2c_projection.launch and enter correct lidar topic, lidar frame_id, and compressed camera topic
##### Enter the name of the file for the LIDAR to CAMERA C_T_L transformation matrix and the projection matrix C_proj_L that  you would like to save - saves a numpy file

    roslaunch calib_lidar_cam calib_l2c_projection.launch

### Terminal 4

    rviz

### Terminal 5

    rosrun rqt_reconfigure rqt_reconfigure

#### Adjust the roll, pitch, yaw, translation x,y, and z value so that the projection of lidar points match the image
#### Save params once satisfied