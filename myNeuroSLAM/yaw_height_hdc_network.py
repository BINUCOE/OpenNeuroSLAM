import numpy as np

# global VT


class YawHeightHDCNetwork:
    def __init__(self, **kwargs):
        
        # The dimension of yaw in yaw_height_hdc network
        self.YAW_HEIGHT_HDC_Y_DIM = kwargs.pop("YAW_HEIGHT_HDC_Y_DIM", 36)

        # The dimension of height in yaw_height_hdc network
        self.YAW_HEIGHT_HDC_H_DIM = kwargs.pop("YAW_HEIGHT_HDC_H_DIM", 36)
        
        # The dimension of local excitation weight matrix for yaw
        self.YAW_HEIGHT_HDC_EXCIT_Y_DIM = kwargs.pop("YAW_HEIGHT_HDC_EXCIT_Y_DIM", 8)

        # The dimension of local excitation weight matrix for height
        self.YAW_HEIGHT_HDC_EXCIT_H_DIM = kwargs.pop("YAW_HEIGHT_HDC_EXCIT_H_DIM", 8)

        # The dimension of local inhibition weight matrix for yaw
        self.YAW_HEIGHT_HDC_INHIB_Y_DIM = kwargs.pop("YAW_HEIGHT_HDC_INHIB_Y_DIM", 5)

        # The dimension of local inhibition weight matrix for height
        self.YAW_HEIGHT_HDC_INHIB_H_DIM = kwargs.pop("YAW_HEIGHT_HDC_INHIB_H_DIM", 5)

        # The global inhibition value
        self.YAW_HEIGHT_HDC_GLOBAL_INHIB = kwargs.pop("YAW_HEIGHT_HDC_GLOBAL_INHIB", 0.0002)

        # amount of energy injected when a view template is re-seen
        self.YAW_HEIGHT_HDC_VT_INJECT_ENERGY = kwargs.pop("YAW_HEIGHT_HDC_VT_INJECT_ENERGY", 0.1)

        # Variance of Excitation and Inhibition in XY and THETA respectively
        self.YAW_HEIGHT_HDC_EXCIT_Y_VAR = kwargs.pop("YAW_HEIGHT_HDC_EXCIT_Y_VAR", 1.9)
        self.YAW_HEIGHT_HDC_EXCIT_H_VAR = kwargs.pop("YAW_HEIGHT_HDC_EXCIT_H_VAR", 1.9)
        self.YAW_HEIGHT_HDC_INHIB_Y_VAR = kwargs.pop("YAW_HEIGHT_HDC_INHIB_Y_VAR", 3.1)
        self.YAW_HEIGHT_HDC_INHIB_H_VAR = kwargs.pop("YAW_HEIGHT_HDC_INHIB_H_VAR", 3.1)

        # The scale of rotation velocity of yaw
        self.YAW_ROT_V_SCALE = kwargs.pop("YAW_ROT_V_SCALE", 1)

        # The scale of rotation velocity of height
        self.HEIGHT_V_SCALE = kwargs.pop("HEIGHT_V_SCALE", 1)

        # packet size for wrap, the left and right activity cells near
        self.YAW_HEIGHT_HDC_PACKET_SIZE = kwargs.pop("YAW_HEIGHT_HDC_PACKET_SIZE", 5)

        # The weight of excitation in yaw_height_hdc network
        self.YAW_HEIGHT_HDC_EXCIT_WEIGHT = self.create_yaw_height_hdc_weights(self.YAW_HEIGHT_HDC_EXCIT_Y_DIM,
                                                                              self.YAW_HEIGHT_HDC_EXCIT_H_DIM,
                                                                              self.YAW_HEIGHT_HDC_EXCIT_Y_VAR,
                                                                              self.YAW_HEIGHT_HDC_EXCIT_H_VAR)

        # The weight of inhibition in yaw_height_hdc network
        self.YAW_HEIGHT_HDC_INHIB_WEIGHT = self.create_yaw_height_hdc_weights(self.YAW_HEIGHT_HDC_INHIB_Y_DIM,
                                                                              self.YAW_HEIGHT_HDC_INHIB_H_DIM,
                                                                              self.YAW_HEIGHT_HDC_INHIB_Y_VAR,
                                                                              self.YAW_HEIGHT_HDC_INHIB_H_VAR)
        # convenience constants
        self.YAW_HEIGHT_HDC_EXCIT_Y_DIM_HALF = int(np.floor(self.YAW_HEIGHT_HDC_EXCIT_Y_DIM / 2))
        self.YAW_HEIGHT_HDC_EXCIT_H_DIM_HALF = int(np.floor(self.YAW_HEIGHT_HDC_EXCIT_H_DIM / 2))
        self.YAW_HEIGHT_HDC_INHIB_Y_DIM_HALF = int(np.floor(self.YAW_HEIGHT_HDC_INHIB_Y_DIM / 2))
        self.YAW_HEIGHT_HDC_INHIB_H_DIM_HALF = int(np.floor(self.YAW_HEIGHT_HDC_INHIB_H_DIM / 2))

        # The size of each unit in radian, 2*pi/ YAW_HEIGHT_HDC_Y_DIM
        self.YAW_HEIGHT_HDC_Y_TH_SIZE = (2 * np.pi) / self.YAW_HEIGHT_HDC_Y_DIM
        self.YAW_HEIGHT_HDC_H_SIZE = (2 * np.pi) / self.YAW_HEIGHT_HDC_H_DIM

        # The lookups for finding the centre of the hdcell in YAW_HEIGHT_HDC by get_hdcell_theta()
        self.YAW_HEIGHT_HDC_Y_SUM_SIN_LOOKUP = np.sin(np.arange(self.YAW_HEIGHT_HDC_Y_DIM) *
                                                      self.YAW_HEIGHT_HDC_Y_TH_SIZE)
        self.YAW_HEIGHT_HDC_Y_SUM_COS_LOOKUP = np.cos(np.arange(self.YAW_HEIGHT_HDC_Y_DIM) *
                                                      self.YAW_HEIGHT_HDC_Y_TH_SIZE)
        self.YAW_HEIGHT_HDC_H_SUM_SIN_LOOKUP = np.sin(np.arange(self.YAW_HEIGHT_HDC_H_DIM) *
                                                      self.YAW_HEIGHT_HDC_H_SIZE)
        self.YAW_HEIGHT_HDC_H_SUM_COS_LOOKUP = np.cos(np.arange(self.YAW_HEIGHT_HDC_H_DIM) *
                                                      self.YAW_HEIGHT_HDC_H_SIZE)
        
        # The excit wrap in yaw_height_hdc network
        self.YAW_HEIGHT_HDC_EXCIT_Y_WRAP = np.concatenate(
            [np.arange(self.YAW_HEIGHT_HDC_Y_DIM - self.YAW_HEIGHT_HDC_EXCIT_Y_DIM_HALF, self.YAW_HEIGHT_HDC_Y_DIM),
                np.arange(self.YAW_HEIGHT_HDC_Y_DIM), np.arange(self.YAW_HEIGHT_HDC_EXCIT_Y_DIM_HALF)])
        self.YAW_HEIGHT_HDC_EXCIT_H_WRAP = np.concatenate(
            [np.arange(self.YAW_HEIGHT_HDC_H_DIM - self.YAW_HEIGHT_HDC_EXCIT_H_DIM_HALF, self.YAW_HEIGHT_HDC_H_DIM),
                np.arange(self.YAW_HEIGHT_HDC_H_DIM), np.arange(self.YAW_HEIGHT_HDC_EXCIT_H_DIM_HALF)])

        # The inhibit wrap in yaw_height_hdc network
        self.YAW_HEIGHT_HDC_INHIB_Y_WRAP = np.concatenate(
            [np.arange(self.YAW_HEIGHT_HDC_Y_DIM - self.YAW_HEIGHT_HDC_INHIB_Y_DIM_HALF, self.YAW_HEIGHT_HDC_Y_DIM),
                np.arange(self.YAW_HEIGHT_HDC_Y_DIM), np.arange(self.YAW_HEIGHT_HDC_INHIB_Y_DIM_HALF)])
        self.YAW_HEIGHT_HDC_INHIB_H_WRAP = np.concatenate(
            [np.arange(self.YAW_HEIGHT_HDC_H_DIM - self.YAW_HEIGHT_HDC_INHIB_H_DIM_HALF, self.YAW_HEIGHT_HDC_H_DIM),
                np.arange(self.YAW_HEIGHT_HDC_H_DIM), np.arange(self.YAW_HEIGHT_HDC_INHIB_H_DIM_HALF)])
        
        # The wrap for finding maximum activity packet
        self.YAW_HEIGHT_HDC_MAX_Y_WRAP = np.concatenate(
            [np.arange(self.YAW_HEIGHT_HDC_Y_DIM - self.YAW_HEIGHT_HDC_PACKET_SIZE, self.YAW_HEIGHT_HDC_Y_DIM),
                np.arange(self.YAW_HEIGHT_HDC_Y_DIM), np.arange(self.YAW_HEIGHT_HDC_PACKET_SIZE)])
        self.YAW_HEIGHT_HDC_MAX_H_WRAP = np.concatenate(
            [np.arange(self.YAW_HEIGHT_HDC_H_DIM - self.YAW_HEIGHT_HDC_PACKET_SIZE, self.YAW_HEIGHT_HDC_H_DIM),
                np.arange(self.YAW_HEIGHT_HDC_H_DIM), np.arange(self.YAW_HEIGHT_HDC_PACKET_SIZE)])

        # set the initial position in the hdcell network
        curYawTheta, curHeight = self.get_hdc_initial_value()
        self.YAW_HEIGHT_HDC = np.zeros((self.YAW_HEIGHT_HDC_Y_DIM, self.YAW_HEIGHT_HDC_H_DIM))
        self.YAW_HEIGHT_HDC[curYawTheta, curHeight] = 1.0

        self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = [curYawTheta, curHeight]
    
    @staticmethod
    def create_yaw_height_hdc_weights(yawDim, heightDim, yawVar, heightVar):
        """Creates a 2D normalised distribution of size dim^2 with a variance of var."""
        # Calculate the center of each dimension
        yawDimCentre = int(np.floor(yawDim / 2))
        heightDimCentre = int(np.floor(heightDim / 2))
        weight = np.zeros((yawDim, heightDim))

        # Fill the weight array with the Gaussian distribution
        for h in range(heightDim):
            for y in range(yawDim):
                weight[y, h] = (1.0 / (yawVar * np.sqrt(2 * np.pi))) * np.exp(
                    -((y - yawDimCentre) ** 2) / (2 * yawVar ** 2)) * (1.0 / (heightVar * np.sqrt(2 * np.pi))) * \
                               np.exp(-((h - heightDimCentre) ** 2) / (2 * heightVar ** 2))

        # Ensure that it is normalised
        total = np.sum(weight)
        weight /= total

        return weight

    @staticmethod
    def get_hdc_initial_value():
        """Set the initial position in the hdcell network."""
        curYawTheta = 0
        curHeight = 0
    
        return curYawTheta, curHeight
    
    def get_current_yaw_height_value(self):
        """
        Returns the approximate averaged centre of the most active activity packet.
        This implementation averages the cells around the maximally activated cell.
        Population Vector Decoding http://blog.yufangwen.com/?p=878
        """
        # Find the max activated cell
        y, h = np.unravel_index(self.YAW_HEIGHT_HDC.argmax(), self.YAW_HEIGHT_HDC.shape, order='F')
    
        # Take the max activated cell +- AVG_CELL in 2d space
        tempYawHeightHdc = np.zeros((self.YAW_HEIGHT_HDC_Y_DIM, self.YAW_HEIGHT_HDC_H_DIM))
        tempYawHeightHdc[np.ix_(self.YAW_HEIGHT_HDC_MAX_Y_WRAP[y: y + self.YAW_HEIGHT_HDC_PACKET_SIZE * 2 + 1],
                         self.YAW_HEIGHT_HDC_MAX_H_WRAP[h: h + self.YAW_HEIGHT_HDC_PACKET_SIZE * 2 + 1])] = \
            self.YAW_HEIGHT_HDC[np.ix_(self.YAW_HEIGHT_HDC_MAX_Y_WRAP[y: y + self.YAW_HEIGHT_HDC_PACKET_SIZE * 2 + 1],
                                self.YAW_HEIGHT_HDC_MAX_H_WRAP[h: h + self.YAW_HEIGHT_HDC_PACKET_SIZE * 2 + 1])]
    
        yawSumSin = np.sum(np.dot(self.YAW_HEIGHT_HDC_Y_SUM_SIN_LOOKUP, np.sum(tempYawHeightHdc, axis=1)))
        yawSumCos = np.sum(np.dot(self.YAW_HEIGHT_HDC_Y_SUM_COS_LOOKUP, np.sum(tempYawHeightHdc, axis=1)))
    
        heightSumSin = np.sum(np.dot(self.YAW_HEIGHT_HDC_H_SUM_SIN_LOOKUP, np.sum(tempYawHeightHdc, axis=0)))
        heightSumCos = np.sum(np.dot(self.YAW_HEIGHT_HDC_H_SUM_COS_LOOKUP, np.sum(tempYawHeightHdc, axis=0)))
    
        outYawTheta = np.mod(np.arctan2(yawSumSin, yawSumCos) / self.YAW_HEIGHT_HDC_Y_TH_SIZE,
                             self.YAW_HEIGHT_HDC_Y_DIM)
        outHeightValue = np.mod(np.arctan2(heightSumSin, heightSumCos) / self.YAW_HEIGHT_HDC_H_SIZE,
                                self.YAW_HEIGHT_HDC_H_DIM)

        return outYawTheta, outHeightValue

    def yaw_height_hdc_iteration(self, vt_id, yawRotV, heightV, VT):
        """
        Pose cell update steps
        1. Add view template energy
        2. Local excitation
        3. Local inhibition
        4. Global inhibition
        5. Normalisation
        6. Path Integration (yawRotV then heightV)
        """
        # If this isn't a new visual template then add the energy at its associated pose cell location
        if VT[vt_id].first != 1:
            act_yaw = int(min(max(np.round(VT[vt_id].hdc_yaw), 1), self.YAW_HEIGHT_HDC_Y_DIM - 1))
            act_height = int(min(max(np.round(VT[vt_id].hdc_height), 1), self.YAW_HEIGHT_HDC_H_DIM - 1))

            energy = self.YAW_HEIGHT_HDC_VT_INJECT_ENERGY * 1.0 / 30.0 * (30.0 - np.exp(1.2 * VT[vt_id].decay))
            if energy > 0:
                self.YAW_HEIGHT_HDC[act_yaw, act_height] += energy
    
        # Local excitation: yaw_height_hdc_local_excitation = yaw_height_hdc elements * yaw_height_hdc weights
        yaw_height_hdc_local_excit_new = np.zeros((self.YAW_HEIGHT_HDC_Y_DIM, self.YAW_HEIGHT_HDC_H_DIM))
        for h in range(self.YAW_HEIGHT_HDC_H_DIM):
            for y in range(self.YAW_HEIGHT_HDC_Y_DIM):
                if self.YAW_HEIGHT_HDC[y, h] != 0:
                    yaw_height_hdc_local_excit_new[
                        np.ix_(self.YAW_HEIGHT_HDC_EXCIT_Y_WRAP[y: y + self.YAW_HEIGHT_HDC_EXCIT_Y_DIM],
                               self.YAW_HEIGHT_HDC_EXCIT_H_WRAP[h: h + self.YAW_HEIGHT_HDC_EXCIT_H_DIM])] += \
                        self.YAW_HEIGHT_HDC[y, h] * self.YAW_HEIGHT_HDC_EXCIT_WEIGHT
    
        self.YAW_HEIGHT_HDC = yaw_height_hdc_local_excit_new
    
        # Local inhibition: yaw_height_hdc_local_inhibition = hdc - hdc elements * hdc_inhib weights
        yaw_height_hdc_local_inhib_new = np.zeros((self.YAW_HEIGHT_HDC_Y_DIM, self.YAW_HEIGHT_HDC_H_DIM))
        for h in range(self.YAW_HEIGHT_HDC_H_DIM):
            for y in range(self.YAW_HEIGHT_HDC_Y_DIM):
                if self.YAW_HEIGHT_HDC[y, h] != 0:
                    yaw_height_hdc_local_inhib_new[
                        np.ix_(self.YAW_HEIGHT_HDC_INHIB_Y_WRAP[y: y + self.YAW_HEIGHT_HDC_INHIB_Y_DIM],
                               self.YAW_HEIGHT_HDC_INHIB_H_WRAP[h: h + self.YAW_HEIGHT_HDC_INHIB_H_DIM])] += \
                        self.YAW_HEIGHT_HDC[y, h] * self.YAW_HEIGHT_HDC_INHIB_WEIGHT
    
        self.YAW_HEIGHT_HDC -= yaw_height_hdc_local_inhib_new
    
        # Global inhibition   PC_gi = PC_li elements - inhibition
        self.YAW_HEIGHT_HDC = np.where(self.YAW_HEIGHT_HDC >= self.YAW_HEIGHT_HDC_GLOBAL_INHIB,
                                       self.YAW_HEIGHT_HDC - self.YAW_HEIGHT_HDC_GLOBAL_INHIB, 0)
    
        # Normalisation
        total = np.sum(self.YAW_HEIGHT_HDC)
        self.YAW_HEIGHT_HDC /= total
    
        # Path integration for yaw
        if yawRotV != 0:
            weight = np.mod(np.abs(yawRotV) / self.YAW_HEIGHT_HDC_Y_TH_SIZE, 1)
            if weight == 0:
                weight = 1.0
                
            shift1 = int(np.sign(yawRotV) * np.floor(
                np.mod(np.abs(yawRotV) / self.YAW_HEIGHT_HDC_Y_TH_SIZE, self.YAW_HEIGHT_HDC_Y_DIM)))
            shift2 = int(np.sign(yawRotV) * np.ceil(
                np.mod(np.abs(yawRotV) / self.YAW_HEIGHT_HDC_Y_TH_SIZE, self.YAW_HEIGHT_HDC_Y_DIM)))
            self.YAW_HEIGHT_HDC = np.roll(self.YAW_HEIGHT_HDC, shift1, axis=0) * (1.0 - weight) + np.roll(
                self.YAW_HEIGHT_HDC, shift2, axis=0) * weight
    
        # Path integration for height
        if heightV != 0:
            weight = np.mod(np.abs(heightV) / self.YAW_HEIGHT_HDC_H_SIZE, 1)
            if weight == 0:
                weight = 1.0

            shift1 = int(np.sign(heightV) * np.floor(
                np.mod(np.abs(heightV) / self.YAW_HEIGHT_HDC_H_SIZE, self.YAW_HEIGHT_HDC_H_DIM)))
            shift2 = int(np.sign(heightV) * np.ceil(
                np.mod(np.abs(heightV) / self.YAW_HEIGHT_HDC_H_SIZE, self.YAW_HEIGHT_HDC_H_DIM)))
            self.YAW_HEIGHT_HDC = np.roll(self.YAW_HEIGHT_HDC, shift1, axis=1) * (1.0 - weight) + np.roll(
                self.YAW_HEIGHT_HDC, shift2, axis=1) * weight

        self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = self.get_current_yaw_height_value()
        return self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH
    