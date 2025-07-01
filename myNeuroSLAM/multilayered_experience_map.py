import numpy as np


class Experience:
    def __init__(self):
        """A point in the experience map"""
        
        self.x_gc = None
        self.y_gc = None
        self.z_gc = None
        self.yaw_hdc = None
        self.height_hdc = None

        self.x_exp = None
        self.y_exp = None
        self.z_exp = None
        self.yaw_exp_rad = None

        self.vt_id = None
        self.numlinks = None
        self.links = []
        

class ExperienceLink:
    """A directed link between two Experience objects"""
    
    def __init__(self):
        self.exp_id = 0
        self.d_xy = 0
        self.d_z = 0
        self.heading_yaw_exp_rad = 0
        self.facing_yaw_exp_rad = 0


class ExperienceMap:
    def __init__(self, **kwargs):
        """Create a new experience and link current experience to it"""
        
        self.GC_X_DIM = kwargs.pop("GC_X_DIM", 36)
        self.GC_Y_DIM = kwargs.pop("GC_Y_DIM", 36)
        self.GC_Z_DIM = kwargs.pop("GC_Z_DIM", 36)
        self.YAW_HEIGHT_HDC_Y_DIM = kwargs.pop("YAW_HEIGHT_HDC_Y_DIM", 36)
        self.YAW_HEIGHT_HDC_H_DIM = kwargs.pop("YAW_HEIGHT_HDC_H_DIM", 36)
        
        # The number of times to run the experience map correction per frame
        self.EXP_LOOPS = kwargs.pop("EXP_LOOPS", 5)
        
        # The amount to correct each experience on either side of a link ( >0.5 is unstable)
        self.EXP_CORRECTION = kwargs.pop("EXP_CORRECTION", 0.5)
        
        # The threshold change in pose cell activity to generate a new experience given the same vt
        self.DELTA_EXP_GC_HDC_THRESHOLD = kwargs.pop("DELTA_EXP_GC_HDC_THRESHOLD", 30)  # The exp delta threshold
        
        # All experiences
        self.EXPERIENCES = []
        
        # The number of total experiences
        self.NUM_EXPS = 0
        
        # The current experience ID
        self.CUR_EXP_ID = 0
        
        # integrate the delta x, y, z, yaw
        self.ACCUM_DELTA_X = 0
        self.ACCUM_DELTA_Y = 0
        self.ACCUM_DELTA_Z = 0
        self.ACCUM_DELTA_YAW = 0
        
        # trajectory of delta of em
        self.DELTA_EM = []
        
        self.PREV_VT_ID = 0
        self.PREV_EXP_ID = 0
        self.exp_initial()
    
    @staticmethod
    def clip_radian_180(angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle <= -np.pi:
            angle += 2 * np.pi
        return angle
    
    @staticmethod
    def clip_radian_360(angle):
        while angle < 0:
            angle += 2 * np.pi
        while angle >= 2 * np.pi:
            angle -= 2 * np.pi
        return angle
    
    @staticmethod
    def get_min_delta(d1, d2, maximum):
        # get the minimum delta distance between two values assuming a wrap to zero at max
        delta = np.min([np.abs(d1 - d2), maximum - np.abs(d1 - d2)])
        return delta
    
    @staticmethod
    def get_signed_delta_radian(angle1, angle2):
        direction = ExperienceMap.clip_radian_180(angle2 - angle1)
        delta_angle = np.abs(ExperienceMap.clip_radian_360(angle1) - ExperienceMap.clip_radian_360(angle2))
        if delta_angle < (2 * np.pi - delta_angle):
            if direction > 0:
                angle = delta_angle
            else:
                angle = -delta_angle
        else:
            if direction > 0:
                angle = 2 * np.pi - delta_angle
            else:
                angle = -(2 * np.pi - delta_angle)
        return angle
    
    def exp_initial(self):
        # Initialize the first experience
        if len(self.EXPERIENCES) == 0:
            self.EXPERIENCES.append(Experience())
        self.EXPERIENCES[0].x_gc = int(np.floor(self.GC_X_DIM / 2)) - 1
        self.EXPERIENCES[0].y_gc = int(np.floor(self.GC_Y_DIM / 2)) - 1
        self.EXPERIENCES[0].z_gc = int(np.floor(self.GC_Z_DIM / 2)) - 1
        self.EXPERIENCES[0].yaw_hdc = 0
        self.EXPERIENCES[0].height_hdc = 0
        self.EXPERIENCES[0].vt_id = 0
        self.EXPERIENCES[0].x_exp = 0
        self.EXPERIENCES[0].y_exp = 0
        self.EXPERIENCES[0].z_exp = 0
        self.EXPERIENCES[0].yaw_exp_rad = 0
        self.EXPERIENCES[0].numlinks = 0
        self.EXPERIENCES[0].links = []
    
    def create_new_exp(self, curExpId, newExpId, vt_id, xGc, yGc, zGc, curYawHdc, curHeight, VT):
        """add link information to the current experience for the new experience
           including the experience_id, odo distance to the experience,
           odo heading (relative to the current experience's facing) to the experience,
           odo delta facing (relative to the current experience's facing)"""
        self.EXPERIENCES[curExpId].numlinks += 1
        new_link = ExperienceLink()
        new_link.exp_id = newExpId
        new_link.d_xy = np.sqrt(self.ACCUM_DELTA_X ** 2 + self.ACCUM_DELTA_Y ** 2)
        new_link.d_z = self.ACCUM_DELTA_Z
        new_link.heading_yaw_exp_rad = self.get_signed_delta_radian(
            self.EXPERIENCES[curExpId].yaw_exp_rad, -np.arctan2(self.ACCUM_DELTA_Y, self.ACCUM_DELTA_X))
        new_link.facing_yaw_exp_rad = self.get_signed_delta_radian(
            self.EXPERIENCES[curExpId].yaw_exp_rad, self.ACCUM_DELTA_YAW)
        self.EXPERIENCES[curExpId].links.append(new_link)
            
        # create the new experience which will have no links to begin with
        # associate with 3d gc
        self.EXPERIENCES.append(Experience())
        self.EXPERIENCES[newExpId].x_gc = xGc
        self.EXPERIENCES[newExpId].y_gc = yGc
        self.EXPERIENCES[newExpId].z_gc = zGc

        # associate with HDC
        self.EXPERIENCES[newExpId].yaw_hdc = curYawHdc
        self.EXPERIENCES[newExpId].height_hdc = curHeight
        
        # associate with VT
        self.EXPERIENCES[newExpId].vt_id = vt_id
        
        # update the coordinate of experience map (x_exp, y_exp, z_exp, yaw_exp_rad, height_exp)
        self.EXPERIENCES[newExpId].x_exp = self.EXPERIENCES[curExpId].x_exp + self.ACCUM_DELTA_X
        self.EXPERIENCES[newExpId].y_exp = self.EXPERIENCES[curExpId].y_exp + self.ACCUM_DELTA_Y
        self.EXPERIENCES[newExpId].z_exp = self.EXPERIENCES[curExpId].z_exp + self.ACCUM_DELTA_Z
        self.EXPERIENCES[newExpId].yaw_exp_rad = self.clip_radian_180(self.ACCUM_DELTA_YAW)
        
        self.EXPERIENCES[newExpId].numlinks = 0
        self.EXPERIENCES[newExpId].links = []

        # add this experience id to the vt for efficient lookup
        VT[vt_id].numExp += 1
        VT[vt_id].exps.append(self.EXPERIENCES[newExpId])
        print("link2VT: ", len(VT[vt_id].exps), VT[vt_id].exps[0] == self.EXPERIENCES[newExpId])
    
    def exp_map_iteration(self, vt_id, transV, yawRotV, heightV, xGc, yGc, zGc, curYawHdc, curHeight, VT):
        # Integrate the delta x, y, z, yaw, height
        self.ACCUM_DELTA_YAW = self.clip_radian_180(self.ACCUM_DELTA_YAW + yawRotV)
        self.ACCUM_DELTA_X += transV * np.cos(self.ACCUM_DELTA_YAW)
        self.ACCUM_DELTA_Y += transV * np.sin(self.ACCUM_DELTA_YAW)
        self.ACCUM_DELTA_Z += heightV

        minDeltaX = self.get_min_delta(self.EXPERIENCES[self.CUR_EXP_ID].x_gc, xGc, self.GC_X_DIM)
        minDeltaY = self.get_min_delta(self.EXPERIENCES[self.CUR_EXP_ID].y_gc, yGc, self.GC_Y_DIM)
        minDeltaZ = self.get_min_delta(self.EXPERIENCES[self.CUR_EXP_ID].z_gc, zGc, self.GC_Z_DIM)
        
        minDeltaYaw = self.get_min_delta(self.EXPERIENCES[self.CUR_EXP_ID].yaw_hdc, curYawHdc,
                                         self.YAW_HEIGHT_HDC_Y_DIM)
        minDeltaHeight = self.get_min_delta(self.EXPERIENCES[self.CUR_EXP_ID].height_hdc, curHeight,
                                            self.YAW_HEIGHT_HDC_H_DIM)
    
        minDeltaYawReversed = self.get_min_delta(self.EXPERIENCES[self.CUR_EXP_ID].yaw_hdc,
                                                 (self.YAW_HEIGHT_HDC_Y_DIM / 2) - curYawHdc, self.YAW_HEIGHT_HDC_Y_DIM)
        minDeltaYaw = min(minDeltaYaw, minDeltaYawReversed)
        
        delta_em = np.sqrt(minDeltaX ** 2 + minDeltaY ** 2 + minDeltaZ ** 2 + minDeltaYaw ** 2 + minDeltaHeight ** 2)
        self.DELTA_EM.append(delta_em)
        # print('$$$', delta_em, vt_id, VT[vt_id].numExp, self.EXPERIENCES[self.CUR_EXP_ID].vt_id, '$$$')
        
        # If the visual template is new or the 3D grid cells (x,y,z) and head direction cells (yaw, height)
        # has changed enough create a new experience
        if VT[vt_id].numExp == 0 or delta_em > self.DELTA_EXP_GC_HDC_THRESHOLD:
            print("create a new experience")
            self.NUM_EXPS += 1
            self.create_new_exp(self.CUR_EXP_ID, self.NUM_EXPS, vt_id, xGc, yGc, zGc, curYawHdc, curHeight, VT)
            self.PREV_EXP_ID = self.CUR_EXP_ID
            self.CUR_EXP_ID = self.NUM_EXPS

            self.ACCUM_DELTA_X = 0
            self.ACCUM_DELTA_Y = 0
            self.ACCUM_DELTA_Z = 0
            self.ACCUM_DELTA_YAW = self.EXPERIENCES[self.CUR_EXP_ID].yaw_exp_rad

        # if the vt has changed (but isn't new) search for the matching experience
        elif vt_id != self.PREV_VT_ID:
            # find the experience associated with the current visual template and that is under the
            # threshold distance to the centre of grid cell and head direction cell activity
            # multiple experiences are under the threshold then don't match (to reduce hash collisions)
            print("search for matching experience")
            matched_exp_id = 0
            matched_exp_count = 0
            delta_em_array = []

            for search_id in range(VT[vt_id].numExp):
                minDeltaYaw = self.get_min_delta(VT[vt_id].exps[search_id].yaw_hdc, curYawHdc,
                                                 self.YAW_HEIGHT_HDC_Y_DIM)
                minDeltaHeight = self.get_min_delta(VT[vt_id].exps[search_id].height_hdc, curHeight,
                                                    self.YAW_HEIGHT_HDC_H_DIM)
        
                delta_em = np.sqrt(self.get_min_delta(VT[vt_id].exps[search_id].x_gc, xGc, self.GC_X_DIM) ** 2 +
                                   self.get_min_delta(VT[vt_id].exps[search_id].y_gc, yGc, self.GC_Y_DIM) ** 2 +
                                   self.get_min_delta(VT[vt_id].exps[search_id].z_gc, zGc, self.GC_Z_DIM) ** 2 +
                                   minDeltaYaw ** 2 + minDeltaHeight ** 2)
                delta_em_array.append(delta_em)
        
                if delta_em < self.DELTA_EXP_GC_HDC_THRESHOLD:
                    matched_exp_count += 1
    
            if matched_exp_count > 1:
                # Hash table collision, keep the previous experience
                pass
    
            else:
                min_delta = min(delta_em_array)
                min_delta_id = delta_em_array.index(min_delta)
                if min_delta < self.DELTA_EXP_GC_HDC_THRESHOLD:
                    matched_exp_id = VT[vt_id].exps[min_delta_id].vt_id
                    # see if the previous experience already has a link to the current experience
                    link_exists = 0
                    for link_id in range(self.EXPERIENCES[self.CUR_EXP_ID].numlinks):
                        if self.EXPERIENCES[self.CUR_EXP_ID].links[link_id].exp_id == matched_exp_id:
                            print("link_exists = True")
                            link_exists = 1
                            break

                    # if we didn't find a link then create the link between current experience
                    # and the experience for the current vt
                    if link_exists == 0:
                        print("link_create = True")
                        self.EXPERIENCES[self.CUR_EXP_ID].numlinks += 1
                        new_link = ExperienceLink()
                        new_link.exp_id = matched_exp_id
                        new_link.d_xy = np.sqrt(self.ACCUM_DELTA_X ** 2 + self.ACCUM_DELTA_Y ** 2)
                        new_link.d_z = self.ACCUM_DELTA_Z
                        new_link.heading_yaw_exp_rad = self.get_signed_delta_radian(self.EXPERIENCES[
                            self.CUR_EXP_ID].yaw_exp_rad, -np.arctan2(self.ACCUM_DELTA_Y, self.ACCUM_DELTA_X))
                        new_link.facing_yaw_exp_rad = self.get_signed_delta_radian(
                            self.EXPERIENCES[self.CUR_EXP_ID].yaw_exp_rad, self.ACCUM_DELTA_YAW)
                        self.EXPERIENCES[self.CUR_EXP_ID].links.append(new_link)

                # if there wasn't an experience with the current vt and grid cell (x y z)
                # and hdcell (yaw, height), then create a new experience
                if matched_exp_id == 0:
                    self.NUM_EXPS += 1
                    self.create_new_exp(self.CUR_EXP_ID, self.NUM_EXPS, vt_id, xGc, yGc, zGc, curYawHdc, curHeight, VT)
                    matched_exp_id = self.NUM_EXPS
                
                self.PREV_EXP_ID = self.CUR_EXP_ID
                self.CUR_EXP_ID = matched_exp_id
            
                self.ACCUM_DELTA_X = 0
                self.ACCUM_DELTA_Y = 0
                self.ACCUM_DELTA_Z = 0
                self.ACCUM_DELTA_YAW = self.EXPERIENCES[self.CUR_EXP_ID].yaw_exp_rad
        
        # Do the experience map correction iteratively for all the links in all the experiences
        # print("Correcting the experience map")
        for _ in range(self.EXP_LOOPS):
            for exp_id in range(self.NUM_EXPS):
                for link_id in range(self.EXPERIENCES[exp_id].numlinks):
                    # experience 0 has a link to experience 1
                    e0 = exp_id
                    e1 = self.EXPERIENCES[e0].links[link_id].exp_id
                    
                    # work out where e0 thinks e1 (x,y) should be based on the stored link information
                    lx = self.EXPERIENCES[e0].x_exp + self.EXPERIENCES[e0].links[link_id].d_xy * np.cos(
                        self.EXPERIENCES[e0].yaw_exp_rad + self.EXPERIENCES[e0].links[link_id].heading_yaw_exp_rad)
                    ly = self.EXPERIENCES[e0].y_exp + self.EXPERIENCES[e0].links[link_id].d_xy * np.sin(
                        self.EXPERIENCES[e0].yaw_exp_rad + self.EXPERIENCES[e0].links[link_id].heading_yaw_exp_rad)
                    lz = self.EXPERIENCES[e0].z_exp + self.EXPERIENCES[e0].links[link_id].d_z
            
                    # correct e0 and e1 (x,y) by equal but opposite amounts a 0.5 correction parameter means that
                    # e0 and e1 will be fully corrected based on e0's link information
                    self.EXPERIENCES[e0].x_exp += (self.EXPERIENCES[e1].x_exp - lx) * self.EXP_CORRECTION
                    self.EXPERIENCES[e0].y_exp += (self.EXPERIENCES[e1].y_exp - ly) * self.EXP_CORRECTION
                    self.EXPERIENCES[e0].z_exp += (self.EXPERIENCES[e1].z_exp - lz) * self.EXP_CORRECTION
            
                    self.EXPERIENCES[e1].x_exp -= (self.EXPERIENCES[e1].x_exp - lx) * self.EXP_CORRECTION
                    self.EXPERIENCES[e1].y_exp -= (self.EXPERIENCES[e1].y_exp - ly) * self.EXP_CORRECTION
                    self.EXPERIENCES[e1].z_exp -= (self.EXPERIENCES[e1].z_exp - lz) * self.EXP_CORRECTION
            
                    # determine the angle between where e0 thinks e1's facing should be based on the link information
                    TempDeltaYawFacing = self.get_signed_delta_radian(
                        (self.EXPERIENCES[e0].yaw_exp_rad + self.EXPERIENCES[e0].links[link_id].facing_yaw_exp_rad),
                        self.EXPERIENCES[e1].yaw_exp_rad)
            
                    # correct e0 and e1 facing by equal but opposite amounts a 0.5 correction parameter means that
                    # e0 and e1 will be fully corrected based on e0's link information
                    self.EXPERIENCES[e0].yaw_exp_rad = self.clip_radian_180(
                        self.EXPERIENCES[e0].yaw_exp_rad + TempDeltaYawFacing * self.EXP_CORRECTION)
                    self.EXPERIENCES[e1].yaw_exp_rad = self.clip_radian_180(
                        self.EXPERIENCES[e1].yaw_exp_rad - TempDeltaYawFacing * self.EXP_CORRECTION)
