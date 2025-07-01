from imresize import *


class VisualTemplate:
    def __init__(self):
        self.id = None
        self.template = []
        self.decay = None
        self.gc_x = None
        self.gc_y = None
        self.gc_z = None
        self.hdc_yaw = None
        self.hdc_pitch = None
        self.first = None
        self.numExp = None
        self.exps = []


class VisualTemplateManager:
    def __init__(self, **kwargs):

        self.GC_X_DIM = kwargs.pop("GC_X_DIM", 36)
        self.GC_Y_DIM = kwargs.pop("GC_Y_DIM", 36)
        self.GC_Z_DIM = kwargs.pop("GC_Z_DIM", 36)
        
        # define a variable of visual template
        self.VT = []
        
        # define the number of visual templates
        self.NUM_VT = 0
        
        # define a variable of previous vt id
        self.PREV_VT_ID = 0
        
        # define history of visual template
        self.VT_HISTORY = []
        self.VT_HISTORY_FIRST = []
        self.VT_HISTORY_OLD = []
        
        self.DIFFS_ALL_IMGS_VTS = []
        
        # define the threshold for vt matching
        self.VT_MATCH_THRESHOLD = 3.4   # This threshold determines whether a new vt is generated
        
        # define the x, y range of visual template img
        self.VT_IMG_CROP_Y_RANGE = kwargs.pop("VT_IMG_CROP_Y_RANGE", slice(0, 270))
        self.VT_IMG_CROP_X_RANGE = kwargs.pop("VT_IMG_CROP_X_RANGE", slice(0, 480))
        
        # define resized range of visual template img for reducing resolution
        self.VT_IMG_RESIZE_X_RANGE = kwargs.pop("VT_IMG_RESIZE_X_RANGE", 48)
        self.VT_IMG_RESIZE_Y_RANGE = kwargs.pop("VT_IMG_RESIZE_Y_RANGE", 27)
        
        # define x, y shift range for template matching
        # This determines how many +- pixels (therefore rotation) will be tested for match
        self.VT_IMG_X_SHIFT = kwargs.pop("VT_IMG_X_SHIFT", 5)
        self.VT_IMG_Y_SHIFT = kwargs.pop("VT_IMG_Y_SHIFT", 3)
        
        # define decay of vt for updating vt
        self.VT_GLOBAL_DECAY = 0.5  # VT_GLOBAL_DECAY is subtracted for all the vt at each time step
        self.VT_ACTIVE_DECAY = 0.5  # VT_ACTIVE_DECAY is added the best matching vt
        
        # define the patch size in x, y for patch normalisation
        self.PATCH_SIZE_Y_K = kwargs.pop("PATCH_SIZE_Y_K", 5)
        self.PATCH_SIZE_X_K = kwargs.pop("PATCH_SIZE_X_K", 5)
   
        # define half offset for reversed matching
        self.VT_IMG_HALF_OFFSET = [0, int(np.floor(self.VT_IMG_RESIZE_X_RANGE * 10 / 2))]
        self.vtPanoramic = 0
        self.VT_STEP = 1
        
        # 执行初始化方法
        self.vt_initial()
    
    def vt_initial(self):
        # Initialize the first visual template
        if len(self.VT) == 0:
            self.VT.append(VisualTemplate())
        self.VT[0].id = 0
        self.VT[0].template = np.zeros((self.VT_IMG_RESIZE_Y_RANGE, self.VT_IMG_RESIZE_X_RANGE))
        self.VT[0].decay = 0.7
        self.VT[0].gc_x = int(np.floor(self.GC_X_DIM / 2)) - 1
        self.VT[0].gc_y = int(np.floor(self.GC_Y_DIM / 2)) - 1
        self.VT[0].gc_z = int(np.floor(self.GC_Z_DIM / 2)) - 1
        self.VT[0].hdc_yaw = 0
        self.VT[0].hdc_height = 0
        self.VT[0].first = 1  # Don't want to inject energy as the vt is being created
        self.VT[0].numExp = 0
        self.VT[0].exps = []

    def vt_compare_segments(self, seg1, seg2, slenY, slenX, cwlY, cwlX):
        """
        Compare two 1D segments and find the minimum difference and corresponding offsets.

        Parameters:
        - seg1: First segment (1D array)
        - seg2: Second segment (1D array)
        - halfOffsetRange: Half offset range for circular shift
        - slenY: Shift length for Y axis
        - slenX: Shift length for X axis
        - cwlY: Center working length for Y axis
        - cwlX: Center working length for X axis

        Returns:
        - offsetY: Minimum offset in Y axis
        - offsetX: Minimum offset in X axis
        - sdif: Minimum difference
        """
        mindiff = 1e7
        minoffsetX = 0
        minoffsetY = 0
    
        # Compare two 1D segments for each offset and sum the absolute difference
        if self.vtPanoramic:
            for halfOffset in self.VT_IMG_HALF_OFFSET:
                seg2 = np.roll(seg2, halfOffset, axis=1)

                for offsetY in range(slenY + 1):
                    for offsetX in range(slenX + 1):
                        cdiff = abs(seg1[offsetY: cwlY, offsetX: cwlX] - seg2[: cwlY - offsetY, : cwlX - offsetX])
                        cdiff = np.sum(cdiff) / (cwlY - offsetY) * (cwlX - offsetX)
                        if cdiff < mindiff:
                            mindiff = cdiff
                            minoffsetX = offsetX
                            minoffsetY = offsetY

                for offsetY in range(1, slenY + 1):
                    for offsetX in range(1, slenX + 1):
                        cdiff = abs(seg1[: cwlY - offsetY, : cwlX - offsetX] - seg2[offsetY: cwlY, offsetX: cwlX])
                        cdiff = np.sum(cdiff) / (cwlY - offsetY) * (cwlX - offsetX)
                        if cdiff < mindiff:
                            mindiff = cdiff
                            minoffsetX = -offsetX
                            minoffsetY = -offsetY

            offsetX = minoffsetX
            offsetY = minoffsetY
            sdif = mindiff
            
        else:
            for offsetY in range(slenY + 1):
                for offsetX in range(slenX + 1):
                    cdiff = abs(seg1[offsetY: cwlY, offsetX: cwlX] - seg2[: cwlY - offsetY, : cwlX - offsetX])
                    cdiff = np.sum(cdiff) / (cwlY - offsetY) * (cwlX - offsetX)
                    # print(cdiff, offsetY, offsetX)
                    if cdiff < mindiff:
                        mindiff = cdiff
                        minoffsetX = offsetX
                        minoffsetY = offsetY
    
            for offsetY in range(1, slenY + 1):
                for offsetX in range(1, slenX + 1):
                    cdiff = abs(seg1[: cwlY - offsetY, : cwlX - offsetX] - seg2[offsetY: cwlY, offsetX: cwlX])
                    cdiff = np.sum(cdiff) / (cwlY - offsetY) * (cwlX - offsetX)
                    if cdiff < mindiff:
                        mindiff = cdiff
                        minoffsetX = -offsetX
                        minoffsetY = -offsetY

            offsetX = minoffsetX
            offsetY = minoffsetY
            sdif = mindiff

        return offsetY, offsetX, sdif
    
    def __getitem__(self, index):
        return self.VT[index]

    def visual_template(self, rawImg, x, y, z, yaw, height):
        # resize the raw image with constraint range
        subImg = rawImg[self.VT_IMG_CROP_Y_RANGE, self.VT_IMG_CROP_X_RANGE]
        vtResizedImg = np.float32(imresize(subImg, (self.VT_IMG_RESIZE_Y_RANGE, self.VT_IMG_RESIZE_X_RANGE)))
        
        # get the size of template image after resized
        ySizeVtImg = self.VT_IMG_RESIZE_Y_RANGE
        xSizeVtImg = self.VT_IMG_RESIZE_X_RANGE
        ySizeNormImg = ySizeVtImg
    
        # define a temp variable for patch normalization
        # extent the dimension of raw image for patch normalization (extVtImg, extension sub image of vtResizedImg)
        extVtImg = np.zeros((ySizeVtImg + self.PATCH_SIZE_Y_K - 1, xSizeVtImg + self.PATCH_SIZE_X_K - 1))
        extVtImg[int((self.PATCH_SIZE_Y_K - 1) / 2): -int((self.PATCH_SIZE_Y_K - 1) / 2),
                 int((self.PATCH_SIZE_X_K - 1) / 2): -int((self.PATCH_SIZE_X_K - 1) / 2)] = vtResizedImg
        
        # patch normalization is applied to compensate for changes in lighting condition
        normVtImg = np.zeros((self.VT_IMG_RESIZE_Y_RANGE, self.VT_IMG_RESIZE_X_RANGE))
        for v in range(ySizeNormImg):
            for u in range(xSizeVtImg):
                patchImg = extVtImg[v: v + self.PATCH_SIZE_Y_K, u: u + self.PATCH_SIZE_X_K]
                normVtImg[v, u] = (vtResizedImg[v, u] - np.mean(patchImg)) / (np.std(patchImg) + 1e-10) / 255.0
        
        # processing the first 20 visual templates, add the image in to vt directly
        print("vt: self.NUM_VT ", self.NUM_VT)
        if self.NUM_VT < 4:
            self.VT[self.NUM_VT].decay -= self.VT_GLOBAL_DECAY
            if self.VT[self.NUM_VT].decay:
                self.VT[self.NUM_VT].decay = 0
            
            self.VT.append(VisualTemplate())
            self.NUM_VT += 1
            self.VT[self.NUM_VT].id = self.NUM_VT
            self.VT[self.NUM_VT].template = normVtImg
            self.VT[self.NUM_VT].decay = self.VT_ACTIVE_DECAY
            self.VT[self.NUM_VT].gc_x = x
            self.VT[self.NUM_VT].gc_y = y
            self.VT[self.NUM_VT].gc_z = z
            self.VT[self.NUM_VT].hdc_yaw = yaw
            self.VT[self.NUM_VT].hdc_height = height
            self.VT[self.NUM_VT].first = 1
            self.VT[self.NUM_VT].numExp = 0
            self.VT[self.NUM_VT].exps = []
            vt_id = self.NUM_VT
            self.VT_HISTORY_FIRST.append(vt_id)

        else:
            MIN_DIFF_CURR_IMG_VTS = [10] * len(self.VT)
            for k in range(1, len(self.VT)):
                self.VT[k].decay -= self.VT_GLOBAL_DECAY
                if self.VT[k].decay < 0:
                    self.VT[k].decay = 0

                min_offset_y, min_offset_x, MIN_DIFF_CURR_IMG_VTS[k] = self.vt_compare_segments(
                    normVtImg, self.VT[k].template, self.VT_IMG_Y_SHIFT, self.VT_IMG_X_SHIFT, ySizeVtImg, xSizeVtImg)

            diff = min(MIN_DIFF_CURR_IMG_VTS)
            diff_id = MIN_DIFF_CURR_IMG_VTS.index(diff)
            # print("diff: ", diff, diff_id == self.PREV_VT_ID)
            
            # if this intensity template doesn't match any of the existing templates
            # then create a new template
            if diff > self.VT_MATCH_THRESHOLD:
                print("create a new template")
                self.VT.append(VisualTemplate())
                self.NUM_VT += 1
                self.VT[self.NUM_VT].id = self.NUM_VT
                self.VT[self.NUM_VT].template = normVtImg
                self.VT[self.NUM_VT].decay = self.VT_ACTIVE_DECAY
                self.VT[self.NUM_VT].gc_x = x
                self.VT[self.NUM_VT].gc_y = y
                self.VT[self.NUM_VT].gc_z = z
                self.VT[self.NUM_VT].hdc_yaw = yaw
                self.VT[self.NUM_VT].hdc_height = height  # Store height information
                self.VT[self.NUM_VT].first = 1
                self.VT[self.NUM_VT].numExp = 0
                self.VT[self.NUM_VT].exps = []
                vt_id = self.NUM_VT
                self.VT_HISTORY_FIRST.append(vt_id)
                
            else:
                vt_id = diff_id
                self.VT[vt_id].decay += self.VT_ACTIVE_DECAY
                if self.PREV_VT_ID != vt_id:
                    self.VT[vt_id].first = 0
                self.VT_HISTORY_OLD.append(vt_id)
    
        self.VT_HISTORY.append(vt_id)
        # print('best template: ', vt_id, self.VT[vt_id])
        
        return vt_id, self.VT
    