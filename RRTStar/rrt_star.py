import math
import os
import sys
import numpy as np
import time
import threading
from scipy.interpolate import interp1d
import concurrent.futures
from collections import deque

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../RRT/")
np.random.seed(25)
try:
    from rrt import RRT
except ImportError:
    raise

show_animation = True

obs_x = 0
obs_y = 0


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self, start, goal, obstacle_list_circle, obstacle_list_square, rand_area,
                 expand_dis=3.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=500,
                 connect_circle_dist=10.0,
                 clearance=0,
                 detection_range=2.0):
        super().__init__(start, goal, obstacle_list_circle, obstacle_list_square,
                         rand_area, expand_dis, path_resolution, goal_sample_rate, max_iter, clearance)
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.detection_range = detection_range

    def planning(self, animation=True, search_until_max_iter=True):
        """
        rrt star path planning

        animation: flag for animation on or off
        search_until_max_iter: search until max iteration for path improving or not
        """
        self.start.theta = 0
        self.start.path_theta = [self.start.theta]
        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list_circle, self.obstacle_list_square, self.clearance):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

            if animation and i % 5 == 0:
                self.draw_graph(rnd)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list_circle, self.obstacle_list_square, self.clearance):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(t_node, self.obstacle_list_circle, self.obstacle_list_square, self.clearance):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        # if expand_dist exists, search vertices in a range no more than expand_dist
        if hasattr(self, 'expand_dis'): 
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x) ** 2 +
                     (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list_circle, self.obstacle_list_square, self.clearance)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                self.node_list[i] = edge_node
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def generate_obstacle_trajectory(self):

        x = np.linspace(0, 10, num=100)
        y = np.abs(np.sin(x**2/9.0)) + 6

        indexes = []
        count = 0
        for (x_coord, y_coord) in zip(x, y):
            if self.check_collision_obstacle(x_coord, y_coord):
                indexes.append(count)
            count += 1

        x_final = x[indexes]
        y_final = y[indexes]
        func_trajectory = interp1d(x_final, y_final, kind='cubic')

        return [x_final, func_trajectory(x_final)]

    def check_collision_obstacle(self, x_coord, y_coord):

        for (ox, oy, size) in self.obstacle_list_circle:
            dx_list = [ox - x_coord]
            dy_list = [oy - y_coord]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size + 0.5) ** 2:
                return False  # collision

        for (lx, ly, rx, ry) in self.obstacle_list_square:

                if (x_coord > (lx - 0.5)) and (x_coord < (rx + 0.5)) and \
                (y_coord < (ry + 0.5)) and (y_coord > (ly - 0.5)):
                    return False

        return True

    def get_obstacle_location(self):
        path = self.generate_obstacle_trajectory()
        # print("OBSTACLE PATHHHH")
        # print(path)
        f = open("obstaclePath.txt", "a+")
        global obs_x
        global obs_y
        for i in range(len(path[0])):
            obs_x = path[0][i]
            obs_y = path[1][i]
            print("Obstacle Path: " + str(obs_x) + ', ' + str(obs_y).format(threading.current_thread().name))
            f.write(str(obs_x) + ',' + str(obs_y) + '\n')
            time.sleep(1)
        f.close()
        # return self.Node(0, 0)
    
    def check_obstacle_in_range(self, current_node, obstacle_node):
        d, _ = self.calc_distance_and_angle(current_node, obstacle_node)
        if d < self.detection_range:
            print("Obstacle within range....")
            new_node = self.steer(current_node, obstacle_node)
            is_path_blocked = self.check_collision(new_node, self.obstacle_list_circle, self.obstacle_list_square, self.clearance)
            if is_path_blocked:
                print("Obstacle visible....")
                return True
        return False
    
    def normalize_angle(self, angle):
        newAngle = np.rad2deg(angle)
        print(newAngle)
        while newAngle <= -180:
            newAngle += 360
        while newAngle > 180:
            newAngle -= 360
        return newAngle
    
    def check_trajectory_collision(self, current_node, obstacle_node, old_obstacle_node):
        threshold_angle = 140
        angle_direction = math.atan2((obstacle_node.y - current_node.y), (obstacle_node.x - current_node.x))
        print("Angle direction ", angle_direction)
        angle_obs = math.atan2((obstacle_node.y - old_obstacle_node.y), (obstacle_node.x - old_obstacle_node.x))
        print("Obstacle velocity angle ", angle_obs)
        angle_robot = math.atan2((current_node.y - current_node.parent.y), (current_node.x - current_node.parent.x))
        print("Robot velocity angle ", angle_robot)
        if abs(self.normalize_angle(angle_direction - angle_robot)) < threshold_angle:
            angle_v_diff = abs(self.normalize_angle(angle_robot - angle_obs))
            if angle_v_diff > 180 - threshold_angle:
                print("Going to collide")
                return True
            elif angle_v_diff < threshold_angle:
                print("Option 1")
                return False
            else:
                print("Option 2")
                return False
        
    def replan_if_path_blocked(self, current_node, obstacle_node, path):
        print("Current location ", current_node.x, current_node.y)
        print("Old obstacle location ", obstacle_node.x, obstacle_node.y)
        time.sleep(1)
        new_obstacle_node = self.Node(obs_x, obs_y)
        print("New obstacle location ", new_obstacle_node.x, new_obstacle_node.y)
        
        if self.check_trajectory_collision(current_node, new_obstacle_node, obstacle_node):
            #TODO
            #Replan algorithm
            print("REPLANNING REQUIRED")
            path = path
        
        return path
    
    def need_for_replan(self, path):
        time.sleep(39)
        final_path = []
        nodes_to_visit = deque(path)
        prev_node = None
        while len(nodes_to_visit) != 0:
            obstacle_node = self.Node(obs_x, obs_y)
            robot = nodes_to_visit.pop()
            final_path.append(robot)
            current_node = self.Node(robot[0], robot[1])
            current_node.parent = prev_node
            print("Robot Path: " + str(current_node.x) + ', ' + str(current_node.y).format(threading.current_thread().name))
            if prev_node is not None:
                print("Robot Parent: " + str(current_node.parent.x) + ', ' + str(current_node.parent.y).format(threading.current_thread().name))
            if self.check_obstacle_in_range(current_node, obstacle_node):
                new_path = self.replan_if_path_blocked(current_node, obstacle_node, nodes_to_visit)
                nodes_to_visit = deque(new_path)
            prev_node = current_node
            time.sleep(1)

        f = open("nodeReplannedPath.txt", "a+")
        for step in final_path:
            f.write(str(step[0]) + ',' + str(step[1]) + ',' + str(step[2]) + '\n')
        f.close()

def main():
    print("Start " + __file__)
    clearance = 0.1
    radius = 0.0

    # ====Search Path with RRT====
    # obstacle_list_circle = [
    #     (5, 5, 1),
    #     (3, 6, 2),
    #     (3, 8, 2),
    #     (3, 10, 2),
    #     (7, 5, 2),
    #     (9, 5, 2),
    #     (8, 10, 1),
    #     (6, 12, 1),
    # ]  # [x,y,size(radius)]

    obstacleList_circle = [
        (2.1, 3.1, 1),
        (7.1, 3.1, 1),
        (5.1, 5.1, 1),
        (7.1, 8.1, 1)
        # (7, 5, 2),
        # (9, 5, 2),
        # (8, 10, 1)
    ]  # [x, y, radius]

    # ======Rectangular Obstacles (bottom left and top right coord) ======#
    obstacleList_square = [
        (0.35, 4.35, 1.85, 5.85),
        (2.35, 7.35, 3.85, 8.85),
        (8.35, 4.35, 9.85, 5.85)
        ]

    # Set Initial parameters
    rrt_star = RRTStar(start=[0, 0],
                       goal=[6, 10],
                       rand_area=[-2, 15],
                       obstacle_list_circle=obstacleList_circle,
                       obstacle_list_square=obstacleList_square,
                       clearance=clearance+radius)
    open('nodePath.txt', 'w').close()
    open('nodeReplannedPath.txt', 'w').close()
    open('obstaclePath.txt', 'w').close()
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     t1 = executor.submit(rrt_star.planning, show_animation)
    #     path = t1.result()
    # path = rrt_star.planning(animation=show_animation)
    # path = [[6, 10, None], [4.935288207869231, 8.167564596777668, 1.03330664866742], [3.399344460620826, 5.590574723827343, 1.216212269993973], [3.0521441331220442, 4.6527837382741986, 2.095447056264107], [4.053965617034213, 2.9217858427318557, 0.6658839055357819], [2.4812235054113425, 1.6862769393531567, 0.5969134076015434], [0, 0, 0]]
    path = [[6, 10, None], [8.295166061988946, 9.647090488213937, 1.7064226267848028], [8.56558783411669, 7.665456802297366, 1.3988459609471349], [8.052275003299025, 4.709698027004978, 1.6670188814098872], [8.340497421582613, 1.72357548471058, 0.8111946307190926], [6.963232050695838, 0.27335477800552366, 0.18204962000928065], [4.996282682866677, -0.08873662536724658, 0.010189555964332997], [1.9964384220953, -0.11930476428721955, -0.05968781681993424], [0, 0, 0]]


    f = open('nodePath.txt', 'r')
    lines = f.readlines()
    x_int = 0
    y_int = 0
    pts = []
    pts.append([x_int, y_int])
    for line in lines:
        points = line.rstrip().split(',')
        print(points)
        pts.append([float(points[0]), float(points[1])])
    
    t1 = threading.Thread(target=rrt_star.get_obstacle_location, name='t1')
    t2 = threading.Thread(target=rrt_star.need_for_replan, args=(path,), name='t2')

    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    if path is None:
        print("Cannot find path")
    else:
        # print("found path!!")
        # f = open('nodePath.txt', 'r')
        # lines = f.readlines()
        # x_int = 0
        # y_int = 0
        # pts = []
        # pts.append([x_int, y_int])
        print(path)
        # Draw final path

        f1 = open('obstaclePath.txt', 'r')
        lines1 = f1.readlines()
        # x_int = 0
        # y_int = 0
        pts_obs = []
        # pts.append([x_int, y_int])
        for line in lines1:
            points = line.rstrip().split(',')
            # print(points)
            pts_obs.append([float(points[0]), float(points[1])])
            
        if show_animation:
            rrt_star.draw_graph()
            plt.plot([x for (x, y, theta) in path], [y for (x, y, theta) in path], '-b')
            # for line in lines:
            #     points = line.rstrip().split(',')
            #     print(points)
                # pts.append([float(points[0]), float(points[1])])
            
            plt.plot(np.asarray(pts)[:,0], np.asarray(pts)[:,1], '-r')
            plt.plot(np.asarray(pts_obs)[:,0], np.asarray(pts_obs)[:,1], '-y')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()
