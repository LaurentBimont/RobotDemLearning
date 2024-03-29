import communication

class RobotiqBaseCModel:
    """Base class (communication protocol agnostic) for sending commands and receiving the status of the Robotic C-Model gripper"""

    def __init__(self):

        # Initiate output message as an empty list
        self.message = []
        self.client = Communication()

        # Note: after the instantiation, a ".client" member must be added to the object

    def verifyCommand(self, command):
        """Function to verify that the value of each variable satisfy its limits."""
        # Verify that each variable is in its correct range
        command.rACT = max(0, command.rACT)
        command.rACT = min(1, command.rACT)

        command.rGTO = max(0, command.rGTO)
        command.rGTO = min(1, command.rGTO)
        command.rATR = max(0, command.rATR)
        command.rATR = min(1, command.rATR)

        command.rPR = max(0,   command.rPR)
        command.rPR = min(255, command.rPR)

        command.rSP = max(0,   command.rSP)
        command.rSP = min(255, command.rSP)

        command.rFR = max(0,   command.rFR)
        command.rFR = min(255, command.rFR)

        # Return the modified command
        return command

    def refreshCommand(self, command):
        """Function to update the command which will be sent during the next sendCommand() call."""

        # Limit the value of each variable
        command = self.verifyCommand(command)

        # Initiate command as an empty lis
        self.message = []

        # Build the command with each output variable
        # To-Do: add verification that all variables are in their authorized range
        self.message.append(
            command.rACT + (command.rGTO << 3) + (command.rATR << 4))
        self.message.append(0)
        self.message.append(0)
        self.message.append(command.rPR)
        self.message.append(command.rSP)
        self.message.append(command.rFR)

    def sendCommand(self):
        """Send the command to the Gripper."""

        self.client.sendCommand(self.message)

    def getStatus(self):

        # Acquire status from the Gripper
        status = self.client.getStatus(6)

        # Message to output
        message = CModel_robot_input()

        # Assign the values to their respective variables
        message.gACT = (status[0] >> 0) & 0x01
        message.gGTO = (status[0] >> 3) & 0x01
        message.gSTA = (status[0] >> 4) & 0x03
        message.gOBJ = (status[0] >> 6) & 0x03
        message.gFLT = status[2]
        message.gPR = status[3]
        message.gPO = status[4]
        message.gCU = status[5]

        return message


class RobotiqCGripper(RobotiqBaseCModel):
    def __init__(self):
        super().__init__()
        self.cur_status = None

    def _status_cb(self, msg):
        self.cur_status = msg

    def is_ready(self):
        return self.cur_status.gSTA == 3 and self.cur_status.gACT == 1

    def is_reset(self):
        return self.cur_status.gSTA == 0 or self.cur_status.gACT == 0

    def is_moving(self):
        return self.cur_status.gGTO == 1 and self.cur_status.gOBJ == 0

    def is_stopped(self):
        return self.cur_status.gOBJ != 0

    def object_detected(self):
        return self.cur_status.gOBJ == 1 or self.cur_status.gOBJ == 2

    def get_fault_status(self):
        return self.cur_status.gFLT

    def get_pos(self):
        po = self.cur_status.gPO
        return np.clip(0.087/(13.-230.)*(po-230.), 0, 0.087)

    def get_req_pos(self):
        pr = self.cur_status.gPR
        return np.clip(0.087/(13.-230.)*(pr-230.), 0, 0.087)

    def is_closed(self):
        return self.cur_status.gPO >= 230

    def is_opened(self):
        return self.cur_status.gPO <= 13

    # in mA
    def get_current(self):
        return self.cur_status.gCU * 0.1

    def reset(self):
        cmd = CModel_robot_input()
        cmd.rACT = 0
        self.cmd_pub.publish(cmd)

    def activate(self, timeout=-1):
        cmd = CModel_robot_input()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = 0
        cmd.rSP = 255
        cmd.rFR = 150
        self.send(cmd)
        try:
            while not self.is_ready():
                time.sleep(0.1)
                return True
        except Exception:
            raise
            return False

    def auto_release(self):
        cmd = CModel_robot_input()
        cmd.rACT = 1
        cmd.rATR = 1
        self.send(cmd)

    ##
    # Goto position with desired force and velocity
    # @param pos Gripper width in meters. [0, 0.087]
    # @param vel Gripper speed in m/s. [0.013, 0.100]
    # @param force Gripper force in N. [30, 100] (not precise)
    def goto(self, pos, vel, force, block=False, timeout=-1):
        cmd = CModel_robot_input()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = int(np.clip((13.-230.)/0.087 * pos + 230., 0, 255))
        cmd.rSP = int(np.clip(255./(0.1-0.013) * (vel-0.013), 0, 255))
        cmd.rFR = int(np.clip(255./(100.-30.) * (force-30.), 0, 255))
        self.send(cmd)
        return True

    def stop(self, block=False, timeout=-1):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rGTO = 0
        self.send(cmd)

        return True

    def open(self, vel=0.1, force=100, block=False, timeout=-1):
        if self.is_opened():
            return True
        return self.goto(1.0, vel, force, block=block, timeout=timeout)

    def close(self, vel=0.1, force=100, block=False, timeout=-1):
        if self.is_closed():
            return True
        return self.goto(-1.0, vel, force, block=block, timeout=timeout)
