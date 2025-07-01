from imresize import *


class VisualOdometry:
    def __init__(self, **kwargs):
        
        # definition the Y (vertical) range of images for odomentry, including image for
        # translational velocity, image for rotational velocity, and image for pitch velocity
        self.ODO_IMG_TRANS_Y_RANGE = kwargs.pop("ODO_IMG_TRANS_Y_RANGE", slice(0, 270))
        self.ODO_IMG_TRANS_X_RANGE = kwargs.pop("ODO_IMG_TRANS_X_RANGE", slice(0, 480))
        self.ODO_IMG_HEIGHT_V_Y_RANGE = kwargs.pop("ODO_IMG_HEIGHT_V_Y_RANGE", slice(0, 270))
        self.ODO_IMG_HEIGHT_V_X_RANGE = kwargs.pop("ODO_IMG_HEIGHT_V_X_RANGE", slice(0, 480))
        self.ODO_IMG_YAW_ROT_Y_RANGE = kwargs.pop("ODO_IMG_YAW_ROT_Y_RANGE", slice(0, 270))
        self.ODO_IMG_YAW_ROT_X_RANGE = kwargs.pop("ODO_IMG_YAW_ROT_X_RANGE", slice(0, 480))
        
        # define the size of resized images for odo
        self.ODO_IMG_TRANS_RESIZE_RANGE = [130, 240]
        self.ODO_IMG_YAW_ROT_RESIZE_RANGE = [130, 240]
        self.ODO_IMG_HEIGHT_V_RESIZE_RANGE = [130, 240]
        
        # define the scale of translational velocity, rotational velocity, and pitch velocity
        self.ODO_TRANS_V_SCALE = kwargs.pop("ODO_TRANS_V_SCALE", 30)
        self.ODO_YAW_ROT_V_SCALE = kwargs.pop("ODO_YAW_ROT_V_SCALE", 1)
        self.ODO_HEIGHT_V_SCALE = kwargs.pop("ODO_HEIGHT_V_SCALE", 5)
        
        # define the maximum threshold of translational velocity, rotational velocity and pitch velocity
        self.MAX_TRANS_V_THRESHOLD = kwargs.pop("MAX_TRANS_V_THRESHOLD", 0.4)
        self.MAX_YAW_ROT_V_THRESHOLD = kwargs.pop("MAX_YAW_ROT_V_THRESHOLD", 4.2)
        self.MAX_HEIGHT_V_THRESHOLD = kwargs.pop("MAX_HEIGHT_V_THRESHOLD", 0.4)
        
        # define the variable for visual odo shift match in vertical and horizontal
        self.ODO_SHIFT_MATCH_VERT = kwargs.pop("ODO_SHIFT_MATCH_VERT", 30)
        self.ODO_SHIFT_MATCH_HORI = kwargs.pop("ODO_SHIFT_MATCH_HORI", 30)
        
        # define the degree of the field of view in horizontal and vertical
        self.FOV_HORI_DEGREE = kwargs.pop("FOV_HORI_DEGREE", 75)
        self.FOV_VERT_DEGREE = kwargs.pop("FOV_VERT_DEGREE", 20)

        # Initialize sums and previous velocities
        self.PREV_YAW_ROT_V_IMG_X_SUMS = np.zeros(self.ODO_IMG_TRANS_RESIZE_RANGE[1])
        self.PREV_HEIGHT_V_IMG_Y_SUMS = np.zeros(self.ODO_IMG_HEIGHT_V_RESIZE_RANGE[0])
        self.PREV_TRANS_V_IMG_X_SUMS = np.zeros(self.ODO_IMG_TRANS_RESIZE_RANGE[1] - self.ODO_SHIFT_MATCH_HORI)

        # Initialize the previous velocity for keeping stable speed
        self.PREV_TRANS_V = 0.025
        self.PREV_YAW_ROT_V = 0.0
        self.PREV_HEIGHT_V = 0.0
        
        self.DEGREE_TO_RADIAN = np.pi / 180
        self.OFFSET_YAW_ROT = None
        self.OFFSET_HEIGHT_V = None
    
    @staticmethod
    def compare_segments(seg1, seg2, shift_length, compare_length_of_intensity):
        """
        Compare two 1D intensity profiles of the current and
        previous images to find the minimum shift offset and difference

        Parameters:
        - seg1: 1D array of the intensity profile of the current image.
        - seg2: 1D array of the intensity profile of the previous image.
        - shift_length: The range of offsets in pixels to consider.
        - compare_length_of_intensity: The length of the intensity profile to actually compare.

        Returns:
        - out_minimum_offset: The minimum shift offset when the difference of intensity is smallest.
        - out_minimum_difference_intensity: The minimum of intensity profiles.
        """
        minimum_difference_intensity = 1e6
        minimum_offset = 0
        
        for offset in range(shift_length + 1):
            compare_difference_segments = np.abs(seg1[offset: compare_length_of_intensity] -
                                                 seg2[: compare_length_of_intensity - offset])
            sum_compare_difference_segments = sum(compare_difference_segments) / (compare_length_of_intensity - offset)
            if sum_compare_difference_segments <= minimum_difference_intensity:
                minimum_difference_intensity = sum_compare_difference_segments
                minimum_offset = offset
        
        for offset in range(1, shift_length + 1):
            compare_difference_segments = np.abs(seg1[: compare_length_of_intensity - offset] -
                                                 seg2[offset: compare_length_of_intensity])
            sum_compare_difference_segments = sum(compare_difference_segments) / (compare_length_of_intensity - offset)
            if sum_compare_difference_segments <= minimum_difference_intensity:
                minimum_difference_intensity = sum_compare_difference_segments
                minimum_offset = -offset

        return minimum_offset, minimum_difference_intensity

    def visual_odometry(self, rawImg, model=0):
        """ The simple visual odometry with scanline intensity profile algorithm.
            the input is a raw image
            the output including horizontal translational velocity, rotational velocity,
            vertical translational velocity (vertical)"""
        
        # start to compute the horizontal rotational velocity (yaw)
        # get the sub_image for rotational velocity from raw image with range constraint
        subRawImg = rawImg[self.ODO_IMG_YAW_ROT_Y_RANGE, self.ODO_IMG_YAW_ROT_X_RANGE]
        subRawImg = np.float32(imresize(subRawImg, self.ODO_IMG_YAW_ROT_RESIZE_RANGE))
        horiDegPerPixel = self.FOV_HORI_DEGREE / subRawImg.shape[1]
    
        # SUB_YAW_ROT_IMG = subRawImg
        # SUB_TRANS_IMG = subRawImg
    
        # get the x_sum of average sum intensity values in every column of image
        imgXSums = np.sum(subRawImg, axis=0)
        avgIntensity = np.mean(imgXSums)
        imgXSums = imgXSums / avgIntensity
    
        # Compare the current image with the previous image
        minOffsetYawRot, minDiffIntensityRot = self.compare_segments(imgXSums, self.PREV_YAW_ROT_V_IMG_X_SUMS,
                                                                     self.ODO_SHIFT_MATCH_HORI, imgXSums.shape[0])
    
        self.OFFSET_YAW_ROT = minOffsetYawRot
        yawRotV = self.ODO_YAW_ROT_V_SCALE * minOffsetYawRot * horiDegPerPixel  # in deg
        
        if abs(yawRotV) > self.MAX_YAW_ROT_V_THRESHOLD:
            yawRotV = self.PREV_YAW_ROT_V
        else:
            self.PREV_YAW_ROT_V = yawRotV
    
        self.PREV_YAW_ROT_V_IMG_X_SUMS = imgXSums
        self.PREV_TRANS_V_IMG_X_SUMS = imgXSums
    
        # start to compute total translational velocity
        transV = minDiffIntensityRot * self.ODO_TRANS_V_SCALE
    
        if transV > self.MAX_TRANS_V_THRESHOLD:
            transV = self.PREV_TRANS_V
        else:
            self.PREV_TRANS_V = transV
    
        # start to compute the height change velocity
        # get the sub_image for pitch velocity from raw image with range constraint
        subRawImg = rawImg[self.ODO_IMG_HEIGHT_V_Y_RANGE, self.ODO_IMG_HEIGHT_V_X_RANGE]
        subRawImg = np.float32(imresize(subRawImg, self.ODO_IMG_HEIGHT_V_RESIZE_RANGE))
        # vertDegPerPixel = self.FOV_VERT_DEGREE / subRawImg.shape[0]
    
        if minOffsetYawRot >= 0:
            subRawImg = subRawImg[:, minOffsetYawRot:]
        else:
            subRawImg = subRawImg[:, : minOffsetYawRot]
    
        # SUB_HEIGHT_V_IMG = subRawImg
        imageYSums = np.sum(subRawImg, axis=1)
        avgIntensity = np.mean(imageYSums)
        imageYSums = imageYSums / avgIntensity
    
        minOffsetHeightV, minDiffIntensityHeight = self.compare_segments(imageYSums, self.PREV_HEIGHT_V_IMG_Y_SUMS,
                                                                         self.ODO_SHIFT_MATCH_VERT, imageYSums.shape[0])
    
        if minOffsetHeightV < 0:
            minDiffIntensityHeight = -minDiffIntensityHeight
        self.OFFSET_HEIGHT_V = minOffsetHeightV
        
        # Visual Odometry
        if model == 0:
            if minOffsetHeightV > 3:
                # heightV = self.ODO_HEIGHT_V_SCALE * minDiffIntensityHeight
                heightV = 0
            else:
                heightV = 0
            # heightV = 0
        
        # visual_odometry_up
        elif model == 1:
            if minOffsetHeightV > 0:
                heightV = self.ODO_HEIGHT_V_SCALE * minDiffIntensityHeight
            elif self.PREV_HEIGHT_V > 0:
                heightV = self.PREV_HEIGHT_V
            else:
                heightV = 0
            
        # visual_odometry_down
        else:
            if minOffsetHeightV < 0:
                heightV = self.ODO_HEIGHT_V_SCALE * minDiffIntensityHeight
            elif self.PREV_HEIGHT_V < 0:
                heightV = self.PREV_HEIGHT_V
            else:
                heightV = 0
    
        if abs(heightV) > self.MAX_HEIGHT_V_THRESHOLD:
            heightV = self.PREV_HEIGHT_V
        else:
            self.PREV_HEIGHT_V = heightV

        self.PREV_HEIGHT_V_IMG_Y_SUMS = imageYSums
        
        # print("visual_odo: ", f"model={model}", [transV, yawRotV, heightV])
        return transV, yawRotV, heightV
    