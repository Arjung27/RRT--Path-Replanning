#!/usr/bin/env python
import numpy as np 
import rospy
from geometry_msgs.msg import Twist
import os
import time
from std_msgs.msg import Int8

def get_going():
    
    dirpath = os.path.dirname(os.path.realpath(__file__))
    
    f_list = None
    with open(dirpath + '/nodePath.txt', "r") as file1:
        f_list = [float(i) for line in file1 for i in line.split(',') if i.strip()]

    f_list = np.reshape(f_list, (len(f_list)/3,3))
    
    th = f_list[:,2]
    pos = f_list[:,0:2]

    del_pos = []
    del_theta = []

    step_angle = 5
    step_angle *= np.pi/180
    for i in range (0,len(pos)-1,1):
        dp = np.linalg.norm(pos[i+1]-pos[i],axis=0)
        dt = (th[i+1]-th[i])
        if dt < step_angle: # if there is no change in theta
        # if !dt
            del_pos.append(dp)
            del_theta.append(dt)
        else: 
            k_prev = th[i]
            # for k in range(th[i], th[i+1], step_angle):
            while k_prev < th[i+1] and th[i+1]-k_prev > step_angle:

                k = k_prev + step_angle
                del_pos.append(0) #linear vel will be 0 
                del_theta.append(step_angle)
                k_prev = k
            del_pos.append(0) #linear vel will be 0 
            del_theta.append(th[i+1]-k_prev)

    del_pos = np.reshape(del_pos,(len(del_pos),1))
    del_theta = np.reshape(del_theta,(len(del_theta),1))

    del_vals = np.hstack((del_pos,del_theta))

    t_stamp = 1.5 #sec
    velocities = np.asarray(del_vals) * float(1/t_stamp) 
    linear = velocities[:,0]
    angular = velocities[:,1]

    cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=1000)

    move_cmd_init = Twist()
    move_cmd_init.linear.x = 0
    move_cmd_init.angular.z = 0

    cmd_vel.publish(move_cmd_init)

    time.sleep(3)

    cnt = 0
    move_cmd = Twist()

    while not rospy.is_shutdown():

        if cnt < len(linear):
            print("linear {}".format(move_cmd.linear.x))
            print("angular {}".format(move_cmd.angular.z))
            move_cmd.linear.x = linear[cnt]
            move_cmd.angular.z = angular[cnt]
            cnt += 1

            t0 = rospy.Time.now().to_sec()
            tf = t0

            r = rospy.Rate(10)

            while( tf - t0 <= t_stamp):
                cmd_vel.publish(move_cmd)
                tf = rospy.Time.now().to_sec()
                r.sleep()
                
            cmd_vel.publish(move_cmd_init)

if __name__ == '__main__':

    rospy.init_node('vel_publish', anonymous=True)
    get_going()
   
    

