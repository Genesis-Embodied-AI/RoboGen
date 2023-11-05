import os.path as osp
import pybullet as p
import math
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

from my_planar_robot import MyPlanarRobot
import pb_ompl

class Maze2D():
    def __init__(self):
        self.obstacles = []

        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)

        # load robot
        robot_id = p.loadURDF("models/planar_robot_4_link.xacro", (0,0,0))
        robot = MyPlanarRobot(robot_id)
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("RRT")

        # add obstacles
        self.add_obstacles()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def add_obstacles(self):
        obstacles = []

        # add outer wall
        wall1 = self.add_box([5, 0, 1], [0.1, 5, 1])
        wall2 = self.add_box([-5, 0, 1], [0.1, 5, 1])
        wall3 = self.add_box([0, 5, 1], [5, 0.1, 1])
        wall4 = self.add_box([0, -5, 1], [5, 0.1, 1])

        # add inner walls
        # gap positions
        gap1_pos = [1, 3]
        gap2_pos = [-2, -2]
        gap3_pos = [2, -2]
        wall5 = self.add_box([gap1_pos[0], 5 - (4.5 - gap1_pos[1]) / 2, 1], [0.1, (4.5 - gap1_pos[1]) / 2, 1])
        wall6 = self.add_box([gap1_pos[0], -5 + (4.5 + gap1_pos[1]) / 2, 1], [0.1, (4.5 + gap1_pos[1]) / 2, 1])
        wall7 = self.add_box([-5 + (4.5 + gap2_pos[0]) / 2, gap2_pos[1], 1], [(4.5 + gap2_pos[0]) / 2, 0.1, 1])
        wall8 = self.add_box([5 - (4.5 - gap3_pos[0]) / 2, gap3_pos[1], 1], [(4.5 - gap3_pos[0]) / 2, 0.1, 1])
        wall9 = self.add_box([gap2_pos[0] + 0.5 + (gap3_pos[0] - gap2_pos[0] - 1) / 2, gap3_pos[1], 1], [(gap3_pos[0] - gap2_pos[0] - 1) / 2, 0.1, 1])
        obstacles = [wall1, wall2, wall3, wall4, wall5, wall6, wall7, wall8, wall9]

        # store obstacles
        self.obstacles = obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def add_box(self, box_pos, half_box_size):
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)
        return box_id

    def demo(self):
        start = [0,0,0,0,0,0,0]
        goal = [2,1,math.radians(-90),0,0,0,0]

        self.robot.set_state(start)
        res, path = self.pb_ompl_interface.plan(goal)
        if res:
            self.pb_ompl_interface.execute(path)
        return res, path

if __name__ == '__main__':
    maze = Maze2D()
    maze.demo()