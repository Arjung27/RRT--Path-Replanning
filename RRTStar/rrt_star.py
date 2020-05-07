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
                 expand_dis=5.0,
                 path_resolution=1.0,
                 goal_sample_rate=5,
                 max_iter=300,
                 connect_circle_dist=15.0,
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

        x = np.linspace(1, -1, num=50)
        # y = np.abs(np.sin(x**2/9.0)) + 6
        y = np.sqrt(1 - x**2)

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
        time.sleep(30)
        # path = self.generate_obstacle_trajectory()
        # print("OBSTACLE PATHHHH")
        # print(path)
        f = open("obstaclePathScam.txt", "r")
        lines = f.readlines()
        # x_int = 0
        # y_int = 0
        node_path_replan = []
        # pts.append([x_int, y_int])
        global obs_x
        global obs_y

        for line in lines:
            points = line.rstrip().split(',')
        # for i in range(len(path[0])):
            obs_x = float(points[0])
            obs_y = float(points[1])
            print("Obstacle Path: " + str(obs_x) + ', ' + str(obs_y).format(threading.current_thread().name))
            # f.write(str(obs_x) + ',' + str(obs_y) + '\n')
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
        threshold_angle = 181
        angle_direction = math.atan2((obstacle_node.y - current_node.y), (obstacle_node.x - current_node.x))
        print("Angle direction ", angle_direction)
        angle_obs = math.atan2((obstacle_node.y - old_obstacle_node.y), (obstacle_node.x - old_obstacle_node.x))
        print("Obstacle velocity angle ", angle_obs)
        if current_node.parent is None:
            angle_robot = math.atan2((current_node.y), (current_node.x))
        else:
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

    def find_nodes_for_new_parent(self, current_node, sampling_distance):

        nearby_nodes = []
        for i, n in enumerate(self.node_list):
            if n.x == current_node.x and n.y == current_node.y:
                inds = i
        # inds = self.node_list.index(current_node)
        print("Found index", inds)
        for i, node in enumerate(self.node_list):
            if ((node.x - current_node.x)**2 + (node.y - current_node.y)**2)**0.5 < sampling_distance and \
                ((node.x - current_node.x)**2 + (node.y - current_node.y)**2)**0.5 != 0:
                nearby_nodes.append(node)

        return nearby_nodes, inds


    def make_current_node_parent(self, current_node, all_nearby_nodes):

        for node in all_nearby_nodes:
            node.parent = current_node
            
    def get_nearest_node_index_replan(self, node_list, rnd_node):
        dlist = [(node[0] - rnd_node.x) ** 2 + (node[1] - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    def replan(self, remaining_path, new_obstacle_node, current_node):
        remaining_path = list(remaining_path)
        search_until_max_iter = True
        animation = False
        index = self.get_nearest_node_index_replan(remaining_path, new_obstacle_node)
        intrmediate_goal = remaining_path[index - 1]
        sampling_distance = ((current_node.x - intrmediate_goal[0])**2 + (current_node.y - intrmediate_goal[1])**2)**0.5

        all_nearby_nodes, inds = self.find_nodes_for_new_parent(current_node, sampling_distance)
        self.node_list = all_nearby_nodes
        self.make_current_node_parent(current_node, self.node_list)
        self.node_list.append(current_node)
        for n1 in self.node_list:
            self.rewire(n1, [len(self.node_list) - 1])
            # print(n1.x, n1.y, n1.parent.x, n1.parent.y, sampling_distance, len(self.node_list))

        self.connect_circle_dist = 10.0
        self.expand_dis = 3.0
        self.start = current_node
        self.goal_node = self.Node(intrmediate_goal[0], intrmediate_goal[1])
        self.end.x = intrmediate_goal[0]
        self.end.y = intrmediate_goal[1]
        self.min_rand = -sampling_distance
        self.max_rand = sampling_distance + 1
        for i in range(300):
            # print("Iter:", i, ", number of nodes: {}, {}, {}".format(len(self.node_list), (self.start.x, self.start.y), (self.goal_node.x, self.goal_node.y)))
            rnd = self.get_random_node()
            # print("Random Node: {}, {}, {}".format(rnd.x, rnd.y, sampling_distance))
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
                    print("Path Found")
                    return self.generate_final_course_with_replan(last_index)


        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            # print("Last Index: ", last_index)
            print("Path Found")
            return self.generate_final_course_with_replan(last_index)


    def replan_if_path_blocked(self, current_node, obstacle_node, path):
        print("Current location ", current_node.x, current_node.y)
        print("Old obstacle location ", obstacle_node.x, obstacle_node.y)
        time.sleep(1)
        new_obstacle_node = self.Node(obs_x, obs_y)
        print("New obstacle location ", new_obstacle_node.x, new_obstacle_node.y)
        
        if self.check_trajectory_collision(current_node, new_obstacle_node, obstacle_node):
            #TODO
            #Replan algorithm
            path = self.replan(path, new_obstacle_node, current_node)
        
        print(path)
        return path
    
    def need_for_replan(self, path):
        # time.sleep(39)
        path = self.break_points_final(path)
        final_path = []
        nodes_to_visit = deque(path)
        prev_node = None
        while len(nodes_to_visit) != 0:
            obstacle_node = self.Node(obs_x, obs_y)
            robot = nodes_to_visit.popleft()
            final_path.append(robot)
            current_node = self.Node(robot[0], robot[1])
            current_node.parent = prev_node
            print("Robot Path: " + str(current_node.x) + ', ' + str(current_node.y).format(threading.current_thread().name))
            if prev_node is not None:
                print("Robot Parent: " + str(current_node.parent.x) + ', ' + str(current_node.parent.y).format(threading.current_thread().name))
            if self.check_obstacle_in_range(current_node, obstacle_node):
                new_path = self.replan_if_path_blocked(current_node, obstacle_node, nodes_to_visit)
                
                print("NEW PATH")
                nodes_to_visit = deque(new_path)
            prev_node = current_node
            time.sleep(1)

        f = open("nodeReplannedPath.txt", "a+")
        for step in final_path:
            f.write(str(step[0]) + ',' + str(step[1]) + ',' + str(step[2]) + '\n')
        f.close()

    def calc_angle(self, from_node, to_node):
        dx = to_node[0] - from_node[0]
        dy = to_node[1] - from_node[1]
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta
    
    def break_points_final(self, points):
         path = []
         for i in range(len(points) - 1, 0, -1):
             d, theta = self.calc_angle(points[i], points[i-1])
             breaks = math.floor(d / 0.1)
             break_x = np.linspace(points[i][0], points[i-1][0], breaks)
             break_y = np.linspace(points[i][1], points[i-1][1], breaks)
             for num in range(breaks):
                 path.append([break_x[num], break_y[num], theta])
         return path


def main():
    print("Start " + __file__)
    clearance = 0.1
    radius = 0.354/2

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
        (3.1, 2.1, 1),
        (7.1, 2.1, 1),
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
    rrt_star = RRTStar(start=[1, 2],
                       goal=[7, 5],
                       rand_area=[0, 10],
                       obstacle_list_circle=obstacleList_circle,
                       obstacle_list_square=obstacleList_square,
                       clearance=clearance+radius)
    open('nodePath.txt', 'w').close()
    open('nodeReplannedPath.txt', 'w').close()
    open('obstaclePath.txt', 'w').close()
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     t1 = executor.submit(rrt_star.planning, show_animation)
    #     path = t1.result()
    path = rrt_star.planning(animation=True)
    # path = [[6, 10, None], [4.935288207869231, 8.167564596777668, 1.03330664866742], [3.399344460620826, 5.590574723827343, 1.216212269993973], [3.0521441331220442, 4.6527837382741986, 2.095447056264107], [4.053965617034213, 2.9217858427318557, 0.6658839055357819], [2.4812235054113425, 1.6862769393531567, 0.5969134076015434], [0, 0, 0]]
    # path = [[6, 10, None], [8.295166061988946, 9.647090488213937, 1.7064226267848028], [8.56558783411669, 7.665456802297366, 1.3988459609471349], [8.052275003299025, 4.709698027004978, 1.6670188814098872], [8.340497421582613, 1.72357548471058, 0.8111946307190926], [6.963232050695838, 0.27335477800552366, 0.18204962000928065], [4.996282682866677, -0.08873662536724658, 0.010189555964332997], [1.9964384220953, -0.11930476428721955, -0.05968781681993424], [0, 0, 0]]

    if path is None:
        print("Cannot find path")
    else:
        f = open('nodePath.txt', 'r')
        lines = f.readlines()
        # x_int = 0
        # y_int = 0
        pts = []
        # pts.append([x_int, y_int])
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

def scam():
    print("Start " + __file__)
    clearance = 0.1
    radius = 0.354 / 2
    obstacleList_circle = [
        (3.1, 2.1, 1),
        (7.1, 2.1, 1),
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
    rrt_star = RRTStar(start=[1, 2],
                       goal=[6, 10],
                       rand_area=[0, 10],
                       obstacle_list_circle=obstacleList_circle,
                       obstacle_list_square=obstacleList_square,
                       clearance=clearance + radius)
    open('nodePathScam.txt', 'w').close()
    path = rrt_star.planning(animation=True)
    print(path)

    node_path = [[1.0, 2.0, 1.065345583775938], [1.0509686178892548, 2.0921006644950593, 1.065345583775938], [1.1019372357785093, 2.184201328990118, 1.065345583775938], [1.152905853667764, 2.2763019934851774, 1.065345583775938], [1.2038744715570187, 2.3684026579802366, 1.065345583775938], [1.2548430894462734, 2.4605033224752955, 1.065345583775938], [1.3058117073355282, 2.5526039869703547, 1.065345583775938], [1.3567803252247828, 2.644704651465414, 1.065345583775938], [1.4077489431140375, 2.7368053159604733, 1.065345583775938], [1.4587175610032923, 2.828905980455532, 1.065345583775938], [1.5096861788925469, 2.9210066449505914, 1.065345583775938], [1.5606547967818016, 3.01310730944565, 1.065345583775938], [1.6116234146710564, 3.1052079739407095, 1.065345583775938], [1.662592032560311, 3.1973086384357687, 1.065345583775938], [1.7135606504495657, 3.289409302930828, 1.065345583775938], [1.7645292683388205, 3.3815099674258873, 1.065345583775938], [1.815497886228075, 3.473610631920946, 1.065345583775938], [1.8664665041173296, 3.565711296416005, 1.065345583775938], [1.9174351220065844, 3.657811960911064, 1.065345583775938], [1.9684037398958392, 3.7499126254061235, 1.065345583775938], [1.9684037398958392, 3.7499126254061235, 0.9979113233938145], [2.0264814563384927, 3.8399491218369675, 0.9979113233938145], [2.0845591727811463, 3.929985618267812, 0.9979113233938145], [2.1426368892238004, 4.020022114698656, 0.9979113233938145], [2.200714605666454, 4.1100586111295, 0.9979113233938145], [2.2587923221091075, 4.200095107560344, 0.9979113233938145], [2.316870038551761, 4.290131603991188, 0.9979113233938145], [2.374947754994415, 4.380168100422033, 0.9979113233938145], [2.4330254714370687, 4.470204596852876, 0.9979113233938145], [2.4911031878797223, 4.560241093283721, 0.9979113233938145], [2.549180904322376, 4.650277589714564, 0.9979113233938145], [2.6072586207650295, 4.740314086145409, 0.9979113233938145], [2.665336337207683, 4.830350582576253, 0.9979113233938145], [2.723414053650337, 4.920387079007097, 0.9979113233938145], [2.7814917700929906, 5.010423575437941, 0.9979113233938145], [2.8395694865356442, 5.100460071868785, 0.9979113233938145], [2.8976472029782983, 5.190496568299629, 0.9979113233938145], [2.955724919420952, 5.280533064730474, 0.9979113233938145], [3.0138026358636054, 5.370569561161318, 0.9979113233938145], [3.071880352306259, 5.4606060575921616, 0.9979113233938145], [3.1299580687489126, 5.550642554023006, 0.9979113233938145], [3.188035785191566, 5.6406790504538495, 0.9979113233938145], [3.24611350163422, 5.730715546884694, 0.9979113233938145], [3.3041912180768738, 5.820752043315538, 0.9979113233938145], [3.3622689345195274, 5.910788539746383, 0.9979113233938145], [3.4203466509621814, 6.000825036177226, 0.9979113233938145], [3.478424367404835, 6.09086153260807, 0.9979113233938145], [3.5365020838474885, 6.180898029038914, 0.9979113233938145], [3.594579800290142, 6.270934525469759, 0.9979113233938145], [3.594579800290142, 6.270934525469759, 0.9979113233938148], [3.6505198049345573, 6.357656978365811, 0.9979113233938148], [3.706459809578973, 6.444379431261863, 0.9979113233938148], [3.762399814223388, 6.531101884157915, 0.9979113233938148], [3.8183398188678033, 6.6178243370539676, 0.9979113233938148], [3.8742798235122184, 6.70454678995002, 0.9979113233938148], [3.930219828156634, 6.791269242846072, 0.9979113233938148], [3.9861598328010492, 6.877991695742123, 0.9979113233938148], [4.042099837445464, 6.964714148638176, 0.9979113233938148], [4.09803984208988, 7.051436601534228, 0.9979113233938148], [4.153979846734295, 7.13815905443028, 0.9979113233938148], [4.209919851378711, 7.224881507326332, 0.9979113233938148], [4.265859856023126, 7.311603960222384, 0.9979113233938148], [4.321799860667541, 7.398326413118436, 0.9979113233938148], [4.377739865311956, 7.4850488660144885, 0.9979113233938148], [4.4336798699563715, 7.571771318910541, 0.9979113233938148], [4.489619874600787, 7.658493771806593, 0.9979113233938148], [4.545559879245202, 7.745216224702645, 0.9979113233938148], [4.601499883889618, 7.831938677598697, 0.9979113233938148], [4.657439888534033, 7.918661130494749, 0.9979113233938148], [4.713379893178448, 8.0053835833908, 0.9979113233938148], [4.7693198978228635, 8.092106036286854, 0.9979113233938148], [4.825259902467279, 8.178828489182905, 0.9979113233938148], [4.881199907111694, 8.265550942078958, 0.9979113233938148], [4.937139911756109, 8.35227339497501, 0.9979113233938148], [4.993079916400525, 8.438995847871062, 0.9979113233938148], [5.04901992104494, 8.525718300767114, 0.9979113233938148], [5.104959925689355, 8.612440753663167, 0.9979113233938148], [5.160899930333771, 8.699163206559218, 0.9979113233938148], [5.216839934978186, 8.785885659455271, 0.9979113233938148], [5.272779939622601, 8.872608112351323, 0.9979113233938148], [5.328719944267016, 8.959330565247374, 0.9979113233938148], [5.384659948911432, 9.046053018143427, 0.9979113233938148], [5.440599953555847, 9.13277547103948, 0.9979113233938148], [5.4965399582002625, 9.219497923935531, 0.9979113233938148], [5.552479962844678, 9.306220376831583, 0.9979113233938148], [5.608419967489093, 9.392942829727636, 0.9979113233938148], [5.664359972133508, 9.479665282623687, 0.9979113233938148], [5.720299976777923, 9.56638773551974, 0.9979113233938148], [5.776239981422339, 9.653110188415791, 0.9979113233938148], [5.832179986066754, 9.739832641311844, 0.9979113233938148], [5.88811999071117, 9.826555094207896, 0.9979113233938148], [5.944059995355585, 9.913277547103949, 0.9979113233938148], [6.0, 10.0, 0.9979113233938148]]


    # node_path_replan = [[0.0, 0.0, 1.0303768265243125], [0.052499566880359845, 0.08749927813393309, 1.0303768265243125], [0.10499913376071969, 0.17499855626786617, 1.0303768265243125], [0.15749870064107954, 0.26249783440179925, 1.0303768265243125], [0.20999826752143938, 0.34999711253573235, 1.0303768265243125], [0.26249783440179925, 0.43749639066966545, 1.0303768265243125], [0.3149974012821591, 0.5249956688035985, 1.0303768265243125], [0.3674969681625189, 0.6124949469375316, 1.0303768265243125], [0.41999653504287876, 0.6999942250714647, 1.0303768265243125], [0.4724961019232386, 0.7874935032053978, 1.0303768265243125], [0.5249956688035985, 0.8749927813393309, 1.0303768265243125], [0.5774952356839583, 0.962492059473264, 1.0303768265243125], [0.6299948025643182, 1.049991337607197, 1.0303768265243125], [0.682494369444678, 1.1374906157411302, 1.0303768265243125], [0.7349939363250378, 1.2249898938750632, 1.0303768265243125], [0.7874935032053977, 1.3124891720089964, 1.0303768265243125], [0.8399930700857575, 1.3999884501429294, 1.0303768265243125], [0.8924926369661174, 1.4874877282768624, 1.0303768265243125], [0.9449922038464772, 1.5749870064107956, 1.0303768265243125], [0.997491770726837, 1.6624862845447286, 1.0303768265243125], [1.049991337607197, 1.7499855626786618, 1.0303768265243125], [1.1024909044875568, 1.8374848408125948, 1.0303768265243125], [1.1549904713679167, 1.924984118946528, 1.0303768265243125], [1.2074900382482765, 2.012483397080461, 1.0303768265243125], [1.2599896051286363, 2.099982675214394, 1.0303768265243125], [1.3124891720089962, 2.187481953348327, 1.0303768265243125], [1.364988738889356, 2.2749812314822604, 1.0303768265243125], [1.4174883057697159, 2.3624805096161934, 1.0303768265243125], [1.4699878726500757, 2.4499797877501264, 1.0303768265243125], [1.5224874395304355, 2.5374790658840594, 1.0303768265243125], [1.5749870064107954, 2.624978344017993, 1.0303768265243125], [1.6274865732911552, 2.712477622151926, 1.0303768265243125], [1.679986140171515, 2.7999769002858588, 1.0303768265243125], [1.7324857070518749, 2.8874761784197918, 1.0303768265243125], [1.7849852739322347, 2.9749754565537248, 1.0303768265243125], [1.8374848408125946, 3.062474734687658, 1.0303768265243125], [1.8899844076929544, 3.149974012821591, 1.0303768265243125], [1.9424839745733142, 3.237473290955524, 1.0303768265243125], [1.994983541453674, 3.324972569089457, 1.0303768265243125], [2.047483108334034, 3.4124718472233906, 1.0303768265243125], [2.099982675214394, 3.4999711253573236, 1.0303768265243125], [2.1524822420947536, 3.5874704034912566, 1.0303768265243125], [2.2049818089751136, 3.6749696816251896, 1.0303768265243125], [2.2574813758554733, 3.7624689597591225, 1.0303768265243125], [2.3099809427358333, 3.849968237893056, 1.0303768265243125], [2.362480509616193, 3.937467516026989, 1.0303768265243125], [2.414980076496553, 4.024966794160922, 1.0303768265243125], [2.4674796433769126, 4.112466072294855, 1.0303768265243125], [2.5199792102572727, 4.199965350428788, 1.0303768265243125], [2.5724787771376323, 4.287464628562721, 1.0303768265243125], [2.5724787771376323, 4.287464628562721, 0.9952701333597478], [2.6297710184908185, 4.375770525821627, 0.9952701333597478], [2.6870632598440043, 4.4640764230805345, 0.9952701333597478], [2.7443555011971905, 4.552382320339441, 0.9952701333597478], [2.8016477425503763, 4.640688217598347, 0.9952701333597478], [2.8589399839035625, 4.7289941148572545, 0.9952701333597478], [2.9162322252567483, 4.817300012116161, 0.9952701333597478], [2.9735244666099345, 4.905605909375068, 0.9952701333597478], [3.0308167079631207, 4.9939118066339745, 0.9952701333597478], [3.0881089493163065, 5.082217703892881, 0.9952701333597478], [3.1454011906694928, 5.170523601151788, 0.9952701333597478], [3.2026934320226785, 5.2588294984106945, 0.9952701333597478], [3.2599856733758648, 5.347135395669602, 0.9952701333597478], [3.3172779147290505, 5.435441292928508, 0.9952701333597478], [3.3745701560822368, 5.5237471901874144, 0.9952701333597478], [3.431862397435423, 5.612053087446322, 0.9952701333597478], [3.4891546387886088, 5.700358984705228, 0.9952701333597478], [3.546446880141795, 5.788664881964134, 0.9952701333597478], [3.603739121494981, 5.876970779223042, 0.9952701333597478], [3.661031362848167, 5.965276676481948, 0.9952701333597478], [3.661031362848167, 5.965276676481948, 1.5098389106394354], [3.6680045447539262, 6.079529267983954, 1.5098389106394354], [3.6749777266596855, 6.193781859485959, 1.5098389106394354], [3.6819509085654447, 6.308034450987965, 1.5098389106394354], [3.688924090471204, 6.422287042489971, 1.5098389106394354], [3.695897272376963, 6.536539633991977, 1.5098389106394354], [3.7028704542827224, 6.650792225493983, 1.5098389106394354], [3.7098436361884817, 6.765044816995989, 1.5098389106394354], [3.716816818094241, 6.879297408497994, 1.5098389106394354], [3.72379, 6.99355, 1.5098389106394354], [3.72379, 6.99355, 0.9227610361599027], [3.7870180555555555, 7.0770625, 0.9227610361599027], [3.8502461111111113, 7.160575, 0.9227610361599027], [3.913474166666667, 7.2440875, 0.9227610361599027], [3.9767022222222224, 7.3276, 0.9227610361599027], [4.039930277777778, 7.4111125, 0.9227610361599027], [4.103158333333333, 7.494625, 0.9227610361599027], [4.166386388888889, 7.5781375, 0.9227610361599027], [4.229614444444445, 7.66165, 0.9227610361599027], [4.2928425, 7.7451625, 0.9227610361599027], [4.356070555555556, 7.828675, 0.9227610361599027], [4.419298611111111, 7.9121875, 0.9227610361599027], [4.482526666666667, 7.9957, 0.9227610361599027], [4.545754722222222, 8.0792125, 0.9227610361599027], [4.608982777777777, 8.162725, 0.9227610361599027], [4.672210833333334, 8.2462375, 0.9227610361599027], [4.735438888888889, 8.32975, 0.9227610361599027], [4.798666944444444, 8.4132625, 0.9227610361599027], [4.8618950000000005, 8.496775, 0.9227610361599027], [4.925123055555556, 8.5802875, 0.9227610361599027], [4.988351111111111, 8.6638, 0.9227610361599027], [5.0515791666666665, 8.7473125, 0.9227610361599027], [5.114807222222222, 8.830825, 0.9227610361599027], [5.178035277777778, 8.9143375, 0.9227610361599027], [5.241263333333333, 8.99785, 0.9227610361599027], [5.304491388888889, 9.081362500000001, 0.9227610361599027], [5.367719444444445, 9.164875, 0.9227610361599027], [5.4309475, 9.2483875, 0.9227610361599027], [5.494175555555556, 9.331900000000001, 0.9227610361599027], [5.557403611111111, 9.4154125, 0.9227610361599027], [5.620631666666666, 9.498925, 0.9227610361599027], [5.6838597222222225, 9.582437500000001, 0.9227610361599027], [5.747087777777778, 9.66595, 0.9227610361599027], [5.810315833333333, 9.7494625, 0.9227610361599027], [5.873543888888889, 9.832975000000001, 0.9227610361599027], [5.936771944444445, 9.9164875, 0.9227610361599027], [6.0, 10.0, 0.9227610361599027]]

    f1 = open('obstaclePathScam.txt', 'r')
    lines1 = f1.readlines()
    # x_int = 0
    # y_int = 0
    pts_obs = []
    # pts.append([x_int, y_int])
    for line in lines1:
        points = line.rstrip().split(',')
        # print(points)
        pts_obs.append([float(points[0]), float(points[1])])
    f2 = open('nodeReplannedPath.txt', 'r')
    lines2 = f2.readlines()
    # x_int = 0
    # y_int = 0
    node_path_replan = []
    # pts.append([x_int, y_int])
    for line in lines2:
        points = line.rstrip().split(',')
        # print(points)
        node_path_replan.append([float(points[0]), float(points[1])])
    # print(pts_obs)
    if show_animation:
        rrt_star.draw_graph()
        plt.plot([x for (x, y, theta) in node_path], [y for (x, y, theta) in node_path], '-m')
        # plt.plot([x for (x, y, theta) in path], [y for (x, y, theta) in path], '-b')
        # plt.plot(np.asarray(pts_obs)[:, 0], np.asarray(pts_obs)[:, 1], '-r')
        # for line in lines:
        #     points = line.rstrip().split(',')
        #     print(points)
        # pts.append([float(points[0]), float(points[1])])
          # Need for Mac
        count_robot = 0
        for obs_cnt in range(30):
            plt.plot([node_path_replan[obs_cnt][0], node_path_replan[obs_cnt + 1][0]],
                     [node_path_replan[obs_cnt][1], node_path_replan[obs_cnt + 1][1]], '-y', linewidth=2)
            plt.grid(True)
            plt.pause(0.1)
            count_robot += 1
        
        count_obstacle = 0
        for cnt in range(count_robot, len(node_path_replan)-2):
            plt.plot([node_path_replan[count_robot][0], node_path_replan[count_robot+1][0]], [node_path_replan[count_robot][1], node_path_replan[count_robot+1][1]], '-y', linewidth=2)
            plt.plot([pts_obs[count_obstacle][0], pts_obs[count_obstacle + 1][0]],
                     [pts_obs[count_obstacle][1], pts_obs[count_obstacle + 1][1]], '-r')
            plt.grid(True)
            plt.pause(0.1)
            count_robot += 1
            count_obstacle += 1
            
        for obs_cnt in range(count_obstacle, len(pts_obs) - 2):
            plt.plot([pts_obs[obs_cnt][0], pts_obs[obs_cnt + 1][0]],
                     [pts_obs[obs_cnt][1], pts_obs[obs_cnt + 1][1]], '-r')
            plt.grid(True)
            plt.pause(0.1)
    plt.grid(True)
    plt.pause(0.01)
    plt.show()


if __name__ == '__main__':
    main()
