import math
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
np.random.seed(25)

show_animation = True


class Helper:
    class Node:

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
            self.theta = None
            self.path_theta = []
            
        # def __eq__(self, other):
        #     return self.x == other.x and self.y == other.y

    def __init__(self, start, goal, obstacle_list_circle, obstacle_list_square, rand_area,
                 expand_dis=3.0, path_resolution=0.5, goal_sample_rate=5, max_iter=500, clearance=0):

        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list_circle = obstacle_list_circle
        self.obstacle_list_square = obstacle_list_square
        self.clearance = clearance
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list_circle, self.obstacle_list_square, self.clearance):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list_circle, self.obstacle_list_square, self.clearance):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        
        new_node.theta = theta

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        new_node.path_theta = [new_node.theta]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)
            new_node.path_theta.append(theta)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.path_theta.append(theta)

        new_node.parent = from_node

        return new_node

    def generate_final_course_with_replan(self, goal_ind):
        # f = open("nodePath.txt", "r+")
        path = [[self.end.x, self.end.y, self.end.theta]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.theta])
            node = node.parent
        path.append([node.x, node.y, node.theta])
        return path
    
    def generate_final_course(self, goal_ind):
        # f = open("nodePath.txt", "r+")
        path = [[self.end.x, self.end.y, self.end.theta]]
        node = self.node_list[goal_ind]
        n_expand = math.floor(self.expand_dis / self.path_resolution)
        dx = self.end.x - node.x
        dy = self.end.y - node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        distance = (dx ** 2 + dy ** 2) ** 0.5
        breaks = math.floor(distance / 0.04)
        break_x = np.linspace(self.end.x, node.x, breaks)
        break_y = np.linspace(self.end.y, node.y, breaks)
        for i in range(breaks):
            f = open("nodePath.txt", "r+")
            content = f.read()
            f.seek(0, 0)
            # toWrite = str([self.path_resolution * math.cos(theta), self.path_resolution * math.sin(theta)
            #                   , theta])
            toWrite = str([break_x[i], break_y[i], theta])
            f.write(toWrite[1:len(toWrite) - 1] + '\n' + content)
            f.close()
        while node.parent is not None:
            dx = node.x - node.parent.x
            dy = node.y - node.parent.y
            d = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            distance = (dx ** 2 + dy ** 2)**0.5
            breaks = math.floor(distance/0.04)
            break_x = np.linspace(node.x, node.parent.x, breaks)
            break_y = np.linspace(node.y, node.parent.y, breaks)

    
            for i in range(breaks):
                f = open("nodePath.txt", "r+")
                content = f.read()
                f.seek(0, 0)
                # toWrite = str([self.path_resolution * math.cos(theta), self.path_resolution * math.sin(theta)
                #                   , theta])
                toWrite = str([break_x[i], break_y[i], theta])
                f.write(toWrite[1:len(toWrite) - 1] + '\n' + content)
                f.close()
            path.append([node.x, node.y, node.theta])
            node = node.parent
        path.append([node.x, node.y, node.theta])        

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        np.random.seed(25)
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_rand, self.max_rand),
                            random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        # fig, ax = plt.subplots(1)
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list_circle:
            self.plot_circle(ox, oy, size)

        for (lx, ly, rx, ry) in self.obstacle_list_square:
            self.plot_rect(lx, ly, rx, ry)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_rect(lx, ly, rx, ry, color="-b"):
        width = rx - lx
        height = ry - ly
        rect = patches.Rectangle((lx, ly), width, height, fill=True)
        plt.gca().add_patch(rect)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(node, obstacle_list_circle, obstacle_list_square, clearance):

        if node is None:
            return False

        for (ox, oy, size) in obstacle_list_circle:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size + clearance) ** 2:
                return False  # collision

        for (lx, ly, rx, ry) in obstacle_list_square:

            for (x, y) in zip(node.path_x, node.path_y):
                if (x > (lx - clearance)) and (x < (rx + clearance)) and \
                (y < (ry + clearance)) and (y > (ly - clearance)):
                    return False


        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta
