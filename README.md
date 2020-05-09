# RRT--Path-Replanning
The program requires python 3 and ROS Kinetic to run.<br>
To run the RRT* with replanning, navigate to base directory which consists RRT and RRTStar directories.<br>
On the terminal run<br>
`python3 RRTStar/rrt_star.py`<br>
To change start point, end point change appropriate values at line 498.<br>
The program will generate two files, nodePath.txt which contains the initially RRT* planned trajectory, and robotPath.txt which contains the replanned path.<br>
<br>
To run this simulation in gazebo, place the rrt_star_visualize folder in your catkin_ws/src folder.<br>
Place the newly generated robotPath.txt file in the scripts folder.<br>
Make sure the script files are executable by running in the scripts folder,<br>
`chmod +x vel_obstacle.py`<br>
`chmod +x vel_publish.py`<br>
In your catkin_ws, run the following commands.<br>
`catkin_make`<br>
`source devel\setup.bash`<br>
`roslaunch rrt_star_visualize run_all.launch`