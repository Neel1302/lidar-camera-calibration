#!/usr/bin/env python
''' 
    Purpose: Implements LIDAR PointCloud Projection onto Image Plane using known transform C_T_L and projection matrix.
    Subscribed topics: /pylon_camera_node_center/image_rect/compressed; /velodyne_points; /transforms/C_T_L; /transforms/proj_L2C
    Pulished topic: /projection_img_center

    Project: WATonoBus
    Author: Neel Bhatt
    Do not share, copy, or use without seeking permission from the author
'''

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import pcl
import numpy as np
import cv2
import time

global bridge
bridge = CvBridge()

global count
count = 0

global done_lidar
done_lidar = True

global cam_start
cam_start = time.time()

# global segments_array, cluster_thresholds
# segments_array = np.array([2,4,6,8,10,12])
segments_array = np.array([5,10,15,20,25,30])
# segments_array = np.array([10,20,30,40,80,120])

# Transformation and projection matrix for the camera
def callback_transform(transform_msg):
    global transformation_matrix
    transformation_matrix = np.array(transform_msg.data).reshape(4,4)
    # print(transformation_matrix)

def callback_projection(projection_msg):
    global proj_matrix
    proj_matrix = np.array(projection_msg.data).reshape(3,4)
    # print(proj_matrix)


# Pointcloud Processing
def callback_pointcloud(raw_cloud):
    global image, cam_start, done_lidar
    done_lidar = False
    lidar_start = time.time()
    print("-- \nLidar_start: "+str(lidar_start))
    temp_image = image
    
    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # # # just resize the display not the image so it fits in screen
    # cv2.resizeWindow("image", 2048/2, 1536/2)

    # Convert pointcloud2 msg to np record array (with x,y,z,intensity, and rings fields)
    points_np_record = ros_numpy.point_cloud2.pointcloud2_to_array(raw_cloud) # np.array(...) allows for write access to points_np_record
    points_np_record = points_np_record.copy()

    # Convert np record array to np array (with just x,y,z)
    points_np = np.zeros((points_np_record.flatten().shape[0],3))
    points_np[:,0]=points_np_record['x'].flatten()
    points_np[:,1]=points_np_record['y'].flatten()
    points_np[:,2]=points_np_record['z'].flatten()
    points_np_intensity=points_np_record['intensity'].flatten()
    points_np_ring=points_np_record['ring'].flatten()
    # print(points_np_record['x'].shape,points_np_record['intensity'].shape)
    # print("Raw Points")
    # print(points_np)
    # print("")
    


    # Filtering out LIDAR points - enable this block if the output is too slow (lags)
    # lidar_points = np.insert(points_np,3,[1.0],axis = 1)
    # z_axis_filtered_indices = np.logical_and(lidar_points[:,2] > -10,lidar_points[:,2] < 10)
    # x_axis_filtered_indices = np.logical_and(lidar_points[:,0] > -10,lidar_points[:,0] < 10)
    # valid = np.logical_and(z_axis_filtered_indices,x_axis_filtered_indices)
    
    # # y_axis_filtered_indices = np.logical_and(lidar_points[:,1] > 10,lidar_points[:,1] < 10)
    # # valid = np.logical_and(y_axis_filtered_indices, valid)

    # lidar_points = lidar_points[valid]
    # points_np_intensity=points_np_intensity[valid]
    # points_np_ring=points_np_ring[valid]
    # print(lidar_points.shape)


    # Transformation from LIDAR to Image Plane
    camera_points = np.matmul(proj_matrix,lidar_points.T)
    # print(camera_points.T)
    w_scale_vector = camera_points[2,:]
    # print(w_scale_vector)
    camera_points = camera_points.T/w_scale_vector[:,None]
    #print(camera_points)
    camera_points = camera_points[:,:2]
    # print(camera_points)

    # Depth WRT to LIDAR
    # xyz_points = np.column_stack((camera_points,lidar_points[:,1]))
    
    # Depth WRT to CAMERA
    xyz_points = np.column_stack((camera_points,w_scale_vector))
    # print("\n")
    # print(xyz_points)
    
    # ----- Processing Starts -----
    
    # Contains processed (i.e. filtered based on bbox) lidar points
    filtered_lidar_points_array = np.array([])
    
    # Prevent global values to be changed by another image callback
    temp_xmin,temp_ymin,temp_xmax,temp_ymax = np.array([0,0,1280,720]) 
    
    # Vectorized (c++ backend) bounding box bounds based points filtering
    boolean_x_y_indices = np.logical_and( xyz_points[:,0:2] > np.array([temp_xmin,temp_ymin]), xyz_points[:,0:2] < np.array([temp_xmax,temp_ymax]) )
    boolean_indices = np.logical_and(boolean_x_y_indices[:,0],boolean_x_y_indices[:,1])

    # Gather filtered pointcloud based on bbox bounds into a new np array based on index_list
    filtered_lidar_points = lidar_points[:,0:3][boolean_indices]
    xyz_points = xyz_points[boolean_indices]

    points_np_intensity=points_np_intensity[boolean_indices]
    # points_np_ring=points_np_ring[boolean_indices]

    euclidian_distance_array = np.sqrt(filtered_lidar_points[:,0]**2 + filtered_lidar_points[:,1]**2)

    k = 0
    for point in euclidian_distance_array:
        if point < segments_array[0]:            
            # img[int(xyz_points[k,0]),int(xyz_points[k,1])] = (0,0,255)
            cv2.circle(temp_image, (int(xyz_points[k,0]),int(xyz_points[k,1])), 1, (0,0,255), thickness=1, lineType=8, shift=0)
        elif point < segments_array[1]:
            # img[int(xyz_points[k,0]),int(xyz_points[k,1])] = (0,127,255)
            cv2.circle(temp_image, (int(xyz_points[k,0]),int(xyz_points[k,1])), 1, (0,127,255), thickness=1, lineType=8, shift=0)
        elif point < segments_array[2]:
            # img[int(xyz_points[k,0]),int(xyz_points[k,1])] = (0,255,255)
            cv2.circle(temp_image, (int(xyz_points[k,0]),int(xyz_points[k,1])), 1, (0,255,255), thickness=1, lineType=8, shift=0)
        elif point < segments_array[3]:
            # img[int(xyz_points[k,0]),int(xyz_points[k,1])] = (0,255,0)
            cv2.circle(temp_image, (int(xyz_points[k,0]),int(xyz_points[k,1])), 1, (0,255,0), thickness=1, lineType=8, shift=0)
        elif point < segments_array[4]:
            # img[int(xyz_points[k,0]),int(xyz_points[k,1])] = (255,0,0)
            cv2.circle(temp_image, (int(xyz_points[k,0]),int(xyz_points[k,1])), 1, (255,0,0), thickness=1, lineType=8, shift=0)
        elif point < segments_array[5]:
            # img[int(xyz_points[k,0]),int(xyz_points[k,1])] = (130,0,75)
            cv2.circle(temp_image, (int(xyz_points[k,0]),int(xyz_points[k,1])), 1, (130,0,75), thickness=1, lineType=8, shift=0)
        else:
            # img[int(xyz_points[k,0]),int(xyz_points[k,1])] = (211,0,148)
            cv2.circle(temp_image, (int(xyz_points[k,0]),int(xyz_points[k,1])), 1, (211,0,148), thickness=1, lineType=8, shift=0)
        k += 1
    # cv2.imshow("image", temp_image)
    # cv2.waitKey(1)
    proj_img_msg = bridge.cv2_to_imgmsg(temp_image,encoding='bgr8')
    pub_proj_img.publish(proj_img_msg)

    # Publish Filtered Point Cloud
    pcl_cloud = pcl.PointCloud(np.array(filtered_lidar_points, dtype=np.float32))
    
    # Convert PCL object to np array
    new_points_np = pcl_cloud.to_array()

    # Update original np record array with the new np array
    
    # print("Before:",points_np_record.shape)
    points_np_record = np.resize(points_np_record,new_points_np[:,0].shape)
    # print("After:",points_np_record.shape)
    points_np_record['x']=new_points_np[:,0]
    points_np_record['y']=new_points_np[:,1]
    points_np_record['z']=new_points_np[:,2]
    points_np_record['intensity']=points_np_intensity
    # points_np_record['ring']=points_np_ring

    # Convert np record array to Pointcloud2 msg
    new_cloud = ros_numpy.point_cloud2.array_to_pointcloud2(points_np_record, stamp = raw_cloud.header.stamp, frame_id=rospy.get_param("~lidar_frame_id"))

    # Published new cloud
    pub_filtered_cloud.publish(new_cloud)

    lidar_end = time.time()
    print("LIDAR Took:" + str((lidar_end-lidar_start)*1000))
    print("TOTAL Took:" + str((lidar_end-cam_start)*1000) + "\n--")
    done_lidar = True

# Camera Image Processing
def callback_Image(ros_image):
    global image, cam_start, done_lidar, count
    cam_start_internal = time.time()
    print("Cam_start_internal: "+str(cam_start_internal))
    if done_lidar:
        cam_start = time.time()
        if rospy.get_param("~cam_transport") == "raw":
            image = bridge.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")
        else:
            image = bridge.compressed_imgmsg_to_cv2(ros_image, desired_encoding="bgr8")
        count += 1
        if count == 2:
            count = 0
            done_lidar = False 
    cam_end = time.time()
    print("Cam Took:" + str((cam_end-cam_start_internal)*1000))

def proj2Img():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'projection_center' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('projection_center', anonymous=True)

    # For Transformation and Projection Matrix
    rospy.Subscriber("/transforms/C_T_L", Float32MultiArray, callback_transform)
    rospy.Subscriber("/transforms/proj_L2C", Float32MultiArray, callback_projection)


    # For Raw Images
    if rospy.get_param("~cam_transport") == "raw":
        print("Using raw images ... at:")
        rospy.Subscriber(rospy.get_param("~cam_topic"), Image, callback_Image)
        print(rospy.get_param("~cam_topic"))

    # For Compressed Images
    else:
        print("Using compressed images ... at:")
        rospy.Subscriber(rospy.get_param("~cam_topic")+"/compressed", CompressedImage, callback_Image)
        print(rospy.get_param("~cam_topic")+"/compressed")
    
    # For Velodyne LIDAR
    rospy.Subscriber(rospy.get_param("~lidar_topic"), PointCloud2, callback_pointcloud)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    pub_proj_img = rospy.Publisher('/projection_img', Image, queue_size=1)
    pub_filtered_cloud = rospy.Publisher('/projection_cloud', PointCloud2, queue_size=1)
    proj2Img()