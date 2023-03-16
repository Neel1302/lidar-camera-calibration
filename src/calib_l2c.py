#!/usr/bin/env python
''' 
    Purpose: Computes and publishes transformation matix C_T_L and projection matrix for LIDAR points to Image Plane - LIDAR Camera Extrinsics.
    It does calibration via relying on the user to provide rough translations and orientation via measurement/intuition and thereafter visualize
    the projection of the pointcloud onto the image plane with live dynamic reconfigure server for all 6 DOF for fine tuning enabling accurate
    projection
    
    Subscribed topics: None
    Pulished topics: /transforms/C_T_L /transforms/proj_L2C

    Project: WATonoBus
    Author: Neel Bhatt
	Do not share, copy, or use without seeking permission from the author
'''

import rospy
import numpy as np
import getpass
from math import sin,cos,pi

from dynamic_reconfigure.server import Server
from calib_lidar_cam.cfg import calibConfig

from std_msgs.msg import Float32MultiArray

import tf
from tf.transformations import quaternion_from_euler
tf_broadcast = tf.TransformBroadcaster()

import time

#Set print options
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def rot_matrix(roll,pitch,yaw):
	''' Compute Rotation Matrix from {B} to {A} = A_R_B given RPY angles using {A} as fixed axis about which RPY of {B} is given:
		Roll is about x axis, Pitch about y axis, and Yaw about z axis.
		
		Inputs: Roll, pitch, and yaw angles in degrees
		Outputs: A_R_B (3x3)
	'''

	alpha = yaw*pi/180; beta = pitch*pi/180; gamma = roll*pi/180
	Rz = np.array([[cos(alpha), -sin(alpha), 0],[sin(alpha),cos(alpha),0],[0,0,1]])
	Ry = np.array([[cos(beta), 0, sin(beta)],[0,1,0],[-sin(beta),0,cos(beta)]])
	Rx = np.array([[1,0,0],[0,cos(gamma),-sin(gamma)],[0,sin(gamma),cos(gamma)]])
	A_R_B = np.matmul(np.matmul(Rz,Ry),Rx)
	return A_R_B

def compute_transforms(roll_alpha, pitch_beta, yaw_gamma, translation_x, translation_y, translation_z):
	# ----------- Generate Rotation Matrix From C to L, given RPY angles (Fixed Axis: LIDAR) -----------
	L_R_C = rot_matrix(roll_alpha,pitch_beta,yaw_gamma)

	L_p_LC = np.array([[translation_x],[translation_y],[translation_z]])

	rotation_matrix = L_R_C
	translation_vector = L_p_LC

	# ----------- Generate Transformation Matrix From C to L i,e. L_T_C, given Rotation and Translation -----------
	L_T_C = np.hstack((rotation_matrix,translation_vector))
	L_T_C = np.vstack((L_T_C,np.array([0,0,0,1])))

	# ----------- Generate Transformation Matrix From L to C, given L_T_C -----------
	C_T_L = np.linalg.inv(L_T_C)

	transformation_matrix = C_T_L

	# ----------- Generate Projection Matrix From L to Image, given C_T_L and K -----------
	K = np.fromstring(rospy.get_param('~K')[1:-1],dtype=np.float32, sep=',').reshape(3,4)
	projection_matrix = np.matmul(K,C_T_L)

	print("\n-------------- Transformation Matrix --------------\n")
	print(transformation_matrix)
	print("\n-------------- Projection Matrix --------------\n")
	print(projection_matrix)

	# print("\n-------------- Visualize TF in Rviz --------------\n")
	# print("rosrun tf static_transform_publisher "+str(translation_x)+" "+str(translation_y)+" "+str(translation_z)+" "+str(yaw_gamma*pi/180)+" "+str(pitch_beta*pi/180)+" "+str(roll_alpha*pi/180)+" velodyne right_cam_test 50")


	# # Save Transformation and Projection Matrix
	# np.save('transformation_matrix_right.npy', transformation_matrix)
	# np.save('projection_matrix_right.npy', projection_matrix)

	return transformation_matrix, projection_matrix


def callback_dynamic_reconfig(config, level):
    global roll_alpha, pitch_beta, yaw_gamma, translation_x, translation_y, translation_z
    global transformation_matrix,projection_matrix
    # rospy.loginfo("""Reconfiugre Request:\n k:{kernel_size} | minVal:{canny_min} | maxVal:{canny_max}""".format(**config))
    # print(config)
    roll_alpha = config['roll_alpha']
    pitch_beta = config['pitch_beta']
    yaw_gamma = config['yaw_gamma']

    translation_x = config['translation_x']
    translation_y = config['translation_y']
    translation_z = config['translation_z']

    save_transform_flag = config['save_transform']

    print("\n-------------- Roll, Pitch, Yaw, t_x, t_y, t_z  --------------\n")
    print(roll_alpha, pitch_beta, yaw_gamma, translation_x, translation_y, translation_z)

    transformation_matrix,projection_matrix = compute_transforms(roll_alpha, pitch_beta, yaw_gamma, translation_x, translation_y, translation_z)

    transform_pub.publish(Float32MultiArray(data=transformation_matrix.flatten()))
    projection_pub.publish(Float32MultiArray(data=projection_matrix.flatten()))


    if save_transform_flag:    	
    	np.save('/home/'+getpass.getuser()+'/catkin_ws/src/calib_lidar_cam/Results/'+rospy.get_param('~transform_filename')+'.npy',transformation_matrix)
    	np.save('/home/'+getpass.getuser()+'/catkin_ws/src/calib_lidar_cam/Results/'+rospy.get_param('~proj_filename')+'.npy',projection_matrix)

    	print('Saved transformation and projection matrix at '+'/home/'+getpass.getuser()+'/catkin_ws/src/calib_lidar_cam/Results/'+rospy.get_param('~transform_filename')+'.npy')

    	print("\n-------------- Visualize TF in Rviz --------------\n")
    	print("rosrun tf static_transform_publisher "+str(translation_x)+" "+str(translation_y)+" "+str(translation_z)+" "+str(yaw_gamma*pi/180)+" "+str(pitch_beta*pi/180)+" "+str(roll_alpha*pi/180)+" "+rospy.get_param('~transform_from_frame')+" "+rospy.get_param('~transform_to_frame')+" 50")
    	
    	print("\n-------------- Copy to transform_publisher --------------\n")
    	print("<node pkg=\"tf\" type=\"static_transform_publisher\" name=\""+rospy.get_param('~transform_to_frame')+"_tf_broadcaster\" args=\""+str(translation_x)+" "+str(translation_y)+" "+str(translation_z)+" "+str(yaw_gamma*pi/180)+" "+str(pitch_beta*pi/180)+" "+str(roll_alpha*pi/180)+" "+rospy.get_param('~transform_from_frame')+" "+rospy.get_param('~transform_to_frame')+" 50\" />")

    return config


def calib():
	global transform_pub, projection_pub

	rospy.init_node('calib_lidar_cam', anonymous=True)

	transform_pub = rospy.Publisher('/transforms/C_T_L',Float32MultiArray, queue_size=10, latch=True)
	projection_pub = rospy.Publisher('/transforms/proj_L2C',Float32MultiArray, queue_size=10, latch=True)

	srv = Server(calibConfig, callback_dynamic_reconfig)

	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()


if __name__ == '__main__':
    calib()