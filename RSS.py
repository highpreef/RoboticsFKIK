import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


class TM:
    def __init__(self, angle, axis, tfp):
        self.axis = axis
        self.angle = angle
        self.tfp = tfp
        if self.axis == 'X':
            self.rm = np.array([[1, 0, 0],
                                [0, np.cos(self.angle), -np.sin(self.angle)],
                                [0, np.sin(self.angle), np.cos(self.angle)]])
        elif self.axis == 'Y':
            self.rm = np.array([[np.cos(self.angle), 0, np.sin(self.angle)],
                                [0, 1, 0],
                                [-np.sin(self.angle), 0, np.cos(self.angle)]])
        elif self.axis == 'Z':
            self.rm = np.array([[np.cos(self.angle), -np.sin(self.angle), 0],
                                [np.sin(self.angle), np.cos(self.angle), 0],
                                [0, 0, 1]])
        else:
            self.rm = np.eye(3)
        self.p = np.array(tfp)
        self.htm = np.eye(4)
        self.htm[:3, :3] = self.rm
        self.htm[:3, 3] = self.p

    def __str__(self):
        angle = np.rad2deg(self.angle)
        if self.axis == 'X' and self.angle != 0:
            rm = ["[1, 0, 0, ",
                  "[0, cos({}), -sin({}), ".format(angle, angle),
                  "[0, sin({}), cos({}), ".format(angle, angle)]
        elif self.axis == 'Y' and self.angle != 0:
            rm = ["[cos({}), 0, sin({}), ".format(angle, angle),
                  "[0, 1, 0, ",
                  "[-sin({}), 0, cos({}), ".format(angle, angle)]
        elif self.axis == 'Z' and self.angle != 0:
            rm = ["[cos({}), -sin({}), 0, ".format(angle, angle),
                  "[sin({}), cos({}), 0, ".format(angle, angle),
                  "[0, 0, 1, "]
        else:
            rm = ["[1, 0, 0, ", "[0, 1, 0, ", "[0, 0, 1, "]

        return rm[0] + str(self.tfp[0]) + " ]" + "\n" + rm[1] + str(self.tfp[1]) + " ]" + "\n" + rm[2] + str(
            self.tfp[2]) + " ]" + "\n"  + "[0, 0, 0, 1 ]"


class FKIK:
    def __init__(self, joint_angles, joint_axis, joint_translations, mode='DEG'):
        assert(len(joint_angles) == len(joint_axis) == len(joint_translations))
        self.verboseJac = None
        if mode == 'DEG':
            self.joint_angles = list(map(np.deg2rad, joint_angles))
        else:
            self.joint_angles = joint_angles
        self.joint_axis = joint_axis
        self.joint_translations = joint_translations
        self.TMs = self.getTMs()
        self.getJointPoses()
        self.jacobian = None
        self.getJacobian()

    def getTMs(self):
        htms = []
        for angle, axis, translation in zip(self.joint_angles, self.joint_axis, self.joint_translations):
            htms.append(TM(angle, axis, translation))
            #print(angle, axis, translation)
        return htms

    def chain(self):
        print()
        print('####################################')
        print('## Printing The TMs of Each Joint ##')
        print('## evaluating trig exprs          ##')
        print('####################################')
        print()
        print("T world -> L1:")
        print(self.TMs[0].htm)
        print()
        for i in range(1, len(self.TMs) - 1):
            print("T L{} -> L{}:".format(i, i + 1))
            print(self.TMs[i].htm)
            print()
        print("T L{} -> end:".format(i + 1))
        print(self.TMs[-1].htm)
        print()

    def verboseChain(self):
        print()
        print('####################################')
        print('## Printing The TMs of Each Joint ##')
        print('## without evaluating trig exprs  ##')
        print('####################################')
        print()
        print("T world -> L1:")
        print(str(self.TMs[0]))
        print()
        for i in range(1, len(self.TMs) - 1):
            print("T L{} -> L{}:".format(i, i + 1))
            print(str(self.TMs[i]))
            print()
        print("T L{} -> end:".format(i + 1))
        print(str(self.TMs[-1]))
        print()

    def getJointPoses(self):
        self.joint_poses = [self.TMs[0].htm]
        parent = self.TMs[0].htm
        for i in range(1, len(self.TMs)):
            parent = parent @ self.TMs[i].htm
            self.joint_poses.append(parent)

    def jointPoses(self):
        print()
        print('####################################')
        print('## Printing propagated TMs of     ##')
        print('## each joint (pose of joints in  ##')
        print('## the world frame)               ##')
        print('####################################')
        print()
        print("T world -> L1:")
        print(self.joint_poses[0])
        print()
        for i in range(1, len(self.TMs) - 1):
            print("T L{} -> L{}:".format(i, i + 1))
            print(self.joint_poses[i])
            print()
        print("T L{} -> end:".format(i + 1))
        print(self.joint_poses[-1])
        print()

    def plotRobot2D(self, origin=[0,0,0], ori='XZ'):
        assert(len(ori)==2)
        ax1, ax2 = 0 if ori[0] == 'X' else 1 if ori[0] == 'Y' else 2, 0 if ori[1] == 'X' else 1 if ori[1] == 'Y' else 2
        origin = [origin[ax1], origin[ax2]]
        plt.plot(*origin, 'X', lw=10, markersize=15)
        legend = ['World']
        lineX, lineY = [origin[0]], [origin[1]]
        for i, matrix in enumerate(self.joint_poses[1:]):
            plt.plot(matrix[ax1,3], matrix[ax2, 3], 'o')
            lineX.append(matrix[ax1, 3])
            lineY.append(matrix[ax2, 3])
            legend.append('Joint {}'.format(i+1))
        legend[-1] = "endEffector"
        plt.plot(lineX, lineY, '-', lw=0.5)
        plt.legend(legend)
        plt.xlabel(ori[0])
        plt.ylabel(ori[1])
        plt.show()

    def plotRobot3D(self, origin=[0,0,0]):
        ax = plt.axes(projection='3d')
        ax.scatter3D(*origin, s=85)
        legend = ['World']
        lineX, lineY, lineZ = [origin[0]], [origin[1]], [origin[2]]
        for i, matrix in enumerate(self.joint_poses[1:]):
            ax.scatter3D(matrix[0,3], matrix[1, 3], matrix[2,3], 'o')
            lineX.append(matrix[0, 3])
            lineY.append(matrix[1, 3])
            lineZ.append(matrix[2, 3])
            legend.append('Joint {}'.format(i+1))
        legend[-1] = "endEffector"
        ax.plot3D(lineX, lineY, lineZ, '-', lw=0.5)
        ax.legend(legend)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def getJacobian(self):
        positions = []
        rms = []
        j_axis = []
        pos, ori = "[ ", "[ "
        for i, axis in enumerate(self.joint_axis):
            if axis != 'N/A':
                positions.append(self.joint_poses[i][:3,3])
                rms.append(self.joint_poses[i][:3,:3])
                j_axis.append(rms[-1] @ np.array([1,0,0]) if axis == 'X' else rms[-1] @ np.array([0,1,0]) if axis == 'Y' else rms[-1] @ np.array([0,0,1]))
                #j_axis.append(np.array([1, 0, 0]) if axis == 'X' else np.array([0, 1, 0]) if axis == 'Y' else np.array([0, 0, 1]))
        peff = self.joint_poses[-1][:3,3]
        jacobian = []
        for i in range(len(positions)):
            col_pos = np.cross(j_axis[i], peff - positions[i])
            col_ori = j_axis[i]
            pos += "({} X {}).T ".format(j_axis[i], peff - positions[i])
            ori += "({}).T ".format(j_axis[i])
            jacobian.append(np.hstack([col_pos, col_ori]))
        pos += "]"
        ori += "]"
        self.jacobian = np.vstack(jacobian).T
        self.verboseJac = [pos, ori]

    def printJac(self):
        print()
        print('####################################')
        print('##   Printing The Full Jacobian   ##')
        print('####################################')
        print()
        print("J")
        print(self.jacobian)

    def printJacVerbose(self):
        print()
        print('####################################')
        print('## Printing The Full Jacobian     ##')
        print('## without evaluating columns     ##')
        print('####################################')
        print()
        print("J pos")
        print(self.verboseJac[0])
        print()
        print("J ori")
        print(self.verboseJac[1])


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    zero_vec = [0,0,0]
    base_angle, base_axis, base_trans = 0, 'N/A', zero_vec
    eff_angle, eff_axis, eff_trans = 0, 'N/A', [0.1, 0, 0]
    obj = FKIK([base_angle, 0, -45, 45, eff_angle], [base_axis, 'Y', 'Y', 'Y', eff_axis],
               [base_trans, [0, 0, 0.09], [0.035, 0, 0.1], [0.1, 0, 0], eff_trans], mode='DEG')
    obj.verboseChain()
    obj.chain()
    obj.jointPoses()
    obj.printJac()
    obj.printJacVerbose()
    obj.plotRobot2D(ori='XZ', origin=base_trans)
    obj.plotRobot3D(origin=base_trans)

