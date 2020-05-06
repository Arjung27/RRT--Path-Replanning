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
                 expand_dis=30.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=500,
                 connect_circle_dist=50.0,
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

        return func_trajectory(x_final)

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
        #TODO
        # points of spline.
        path = self.generate_obstacle_trajectory()
        
        global obs_x
        global obs_y
        for pts in path:
            obs_x = pts[0]
            obs_y = pts[1]
            time.sleep(1)
        # return self.Node(0, 0)
    
    def check_obstacle_in_range(self, current_node, obstacle_node):
        d, _ = self.calc_distance_and_angle(current_node, obstacle_node)
        if d < self.detection_range:
            new_node = self.steer(current_node, obstacle_node)
            is_path_blocked = self.check_collision(new_node, self.obstacle_list_circle, self.obstacle_list_square, self.clearance)
            if not is_path_blocked:
                return True
        return False
    
    def normalize_angle(self, angle):
        newAngle = angle
        while newAngle <= -180:
            newAngle += 360
        while newAngle > 180:
            newAngle -= 360
        return newAngle
    
    def check_trajectory_collision(self, current_node, obstacle_node, old_obstacle_node):
        threshold_angle = 15
        angle_direction = math.atan2((obstacle_node.y - current_node.y), (obstacle_node.x - current_node.x))
        angle_obs = math.atan2((obstacle_node.y - old_obstacle_node.y), (obstacle_node.x - old_obstacle_node.x))
        angle_robot = math.atan2((current_node.y - current_node.parent.y), (current_node.x - current_node.parent.x))
        if abs(self.normalize_angle(angle_direction - angle_robot)) < threshold_angle:
            angle_v_diff = abs(self.normalize_angle(angle_robot - angle_obs))
            if angle_v_diff < threshold_angle:
                return False
            elif angle_v_diff > 180 - threshold_angle:
                return True
            else:
                return False
        
        
    def replan_if_path_blocked(self, current_node, obstacle_node, path):
        time.sleep(1)
        new_obstacle_node = self.Node(obs_x, obs_y)
        
        if self.check_trajectory_collision(current_node, new_obstacle_node, obstacle_node):
            #TODO
            #Replan algorithm
            path = self.replan()
        
        return path
    
    def need_for_replan(self, path):
        final_path = []
        nodes_to_visit = deque(path)
        prev_node = None
        while len(nodes_to_visit) != 0:
            obstacle_node = self.Node(obs_x, obs_y)
            robot = nodes_to_visit.pop()
            final_path.append(robot)
            current_node = self.Node(robot[0], robot[1])
            current_node.parent = prev_node
            if self.check_obstacle_in_range(current_node, obstacle_node):
                new_path = self.replan_if_path_blocked(current_node, obstacle_node, nodes_to_visit)
                nodes_to_visit = deque(new_path)
            prev_node = current_node
                
        for step in final_path:
            f = open("nodeReplannedPath.txt", "a+")
            # toWrite = str([self.path_resolution * math.cos(theta), self.path_resolution * math.sin(theta)
            #                   , theta])
            toWrite = str([step[0], step[1], step[2]])
            f.write(toWrite[1:len(toWrite) - 1] + '\n')
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
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     t1 = executor.submit(rrt_star.planning, show_animation)
    #     path = t1.result()
    path = rrt_star.planning(animation=show_animation)
    
    t2 = threading.Thread(target=rrt_star.get_obstacle_location)
    t2.start()
    
    t3 = threading.Thread(target=rrt_star.need_for_replan, args=(path,))
    t3.start()
    

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        f = open('nodePath.txt', 'r')
        lines = f.readlines()
        x_int = 0
        y_int = 0
        pts = []
        pts.append([x_int, y_int])
        print(path)
        # Draw final path
        if show_animation:
            rrt_star.draw_graph()
            plt.plot([x for (x, y, theta) in path], [y for (x, y, theta) in path], '-b')
            for line in lines:
                points = line.rstrip().split(',')
                print(points)
                
                # plt.plot(float(points[0]), float(points[1]), '-r')
                # print(points)
                # x_int += float(points[0])
                # y_int += float(points[1])
                pts.append([float(points[0]), float(points[1])])
            
            plt.plot(np.asarray(pts)[:,0], np.asarray(pts)[:,1], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()
