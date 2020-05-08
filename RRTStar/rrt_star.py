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
        f = open("obstaclePath.txt", "r")
        lines = f.readlines()
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
        threshold_angle = 140
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
        print("Current node vals ", current_node.x, current_node.y)
        for i, n in enumerate(self.node_list):
            print(n.x, n.y)
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

    def get_random_node_replan(self, sampling_distance):

        x_coord = np.random.uniform(self.min_rand, self.max_rand)
        y_coord = np.random.uniform(self.min_rand, self.max_rand)

        while (x_coord**2 + y_coord**2)**0.5 >= sampling_distance:
            x_coord = np.random.uniform(self.min_rand, self.max_rand)
            y_coord = np.random.uniform(self.min_rand, self.max_rand)

        return self.Node(x_coord, y_coord)

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
        current_node.parent = None
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
            rnd = self.get_random_node_replan(sampling_distance)
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
                    return [self.generate_final_course_with_replan(last_index), index - 1]


        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            # print("Last Index: ", last_index)
            print("Path Found")
            return [self.generate_final_course_with_replan(last_index), index - 1]


    def replan_if_path_blocked(self, current_node, obstacle_node, path):
        print("Current location ", current_node.x, current_node.y)
        print("Old obstacle location ", obstacle_node.x, obstacle_node.y)
        time.sleep(1)
        new_obstacle_node = self.Node(obs_x, obs_y)
        print("New obstacle location ", new_obstacle_node.x, new_obstacle_node.y)
        new_path = None
        flag = True
        index = None
        if self.check_trajectory_collision(current_node, new_obstacle_node, obstacle_node):
            res = self.replan(path, new_obstacle_node, current_node)
            if res is not None:
                print(res[0])
                new_path = res[0]
                index = res[1]
                flag = False
            # print("Couldn't find new path, continuing with same path")
        
        if new_path is None:
            new_path = path
            flag = True
        return new_path, index, flag
    
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
            current_node = self.Node(float(robot[0]), float(robot[1]))
            current_node.parent = prev_node
            print("Robot Path: " + str(current_node.x) + ', ' + str(current_node.y).format(threading.current_thread().name))
            if prev_node is not None:
                print("Robot Parent: " + str(current_node.parent.x) + ', ' + str(current_node.parent.y).format(threading.current_thread().name))
            if self.check_obstacle_in_range(current_node, obstacle_node):
                new_path, index, flag = self.replan_if_path_blocked(current_node, obstacle_node, nodes_to_visit)
                if not flag:
                    print("NEW PATH")
                    print("Old path ", nodes_to_visit)
                    print("Index ", index)
                    new_path = self.break_points_final(new_path)
                    remaining = list(nodes_to_visit)[index+1:]
                    nodes_to_visit = deque(new_path)
                    for r in remaining:
                        nodes_to_visit.append(r)
                    print("New path ", nodes_to_visit)
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
                 path.append([float(break_x[num]), float(break_y[num]), float(theta)])
         prev = None
         for num, node in enumerate(path):
             new_node = self.Node(node[0], node[1])
             new_node.theta = node[2]
             new_node.parent = prev
             self.node_list.append(new_node)
             prev = node
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
                       goal=[6, 10],
                       rand_area=[0, 10],
                       obstacle_list_circle=obstacleList_circle,
                       obstacle_list_square=obstacleList_square,
                       clearance=clearance+radius)
    open('nodePath.txt', 'w').close()
    open('nodeReplannedPath.txt', 'w').close()
    # open('obstaclePath.txt', 'w').close()
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     t1 = executor.submit(rrt_star.planning, show_animation)
    #     path = t1.result()
    path = rrt_star.planning(animation=True)
    
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

        if show_animation:
            rrt_star.draw_graph()
            plt.plot([x for (x, y, theta) in path], [y for (x, y, theta) in path], '-m')
            count_robot = 0
            img_count = 0
            for obs_cnt in range(30):
                plt.plot([node_path_replan[obs_cnt][0], node_path_replan[obs_cnt + 1][0]],
                         [node_path_replan[obs_cnt][1], node_path_replan[obs_cnt + 1][1]], '-y', linewidth=2)
                plt.savefig('./plots/%04d.png' % img_count)
                img_count += 1
                plt.grid(True)
                plt.pause(0.1)
                count_robot += 1
    
            count_obstacle = 0
            for cnt in range(count_robot, len(node_path_replan) - 2):
                plt.plot([node_path_replan[count_robot][0], node_path_replan[count_robot + 1][0]],
                         [node_path_replan[count_robot][1], node_path_replan[count_robot + 1][1]], '-y', linewidth=2)
                
                if count_obstacle < len(pts_obs) - 1:
                    plt.plot([pts_obs[count_obstacle][0], pts_obs[count_obstacle + 1][0]],
                             [pts_obs[count_obstacle][1], pts_obs[count_obstacle + 1][1]], '-r')
                plt.savefig('./plots/%04d.png' % img_count)
                img_count += 1
                plt.grid(True)
                plt.pause(0.1)
                count_robot += 1
                count_obstacle += 1
    
            for obs_cnt in range(count_obstacle, len(pts_obs) - 2):
                plt.plot([pts_obs[obs_cnt][0], pts_obs[obs_cnt + 1][0]],
                         [pts_obs[obs_cnt][1], pts_obs[obs_cnt + 1][1]], '-r')
                plt.savefig('./plots/%04d.png' % img_count)
                img_count += 1
                plt.grid(True)
                plt.pause(0.1)
            plt.grid(True)
            plt.pause(0.01)
            plt.show()


if __name__ == '__main__':
    main()
