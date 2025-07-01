from PIL import Image
import numpy as np
import mpl_toolkits.mplot3d
import os
import matplotlib.pyplot as plt
import grid_cells_network
import yaw_height_hdc_network
import multilayered_experience_map
import visual_odometry
import visual_template
import time


class NeuroSLAM:
    def __init__(self, image_source_list):
        self.image_source_list = image_source_list
        self.DEGREE_TO_RADIAN = np.pi / 180
        self.RADIAN_TO_DEGREE = 180 / np.pi
        self.KEY_POINT_SET = np.array([3750, 4700, 8193, 9210])
        self.ODO_STEP = 5


def load_image(image_path):
    img_list = []
    for img_name in os.listdir(image_path):
        img_list.append(image_path + img_name)
    
    return np.sort(img_list)


def rgb2gray(rgb_image):
    """
    Convert an RGB image to a grayscale image.

    Parameters:
    - rgb_image: A 3D numpy array representing the RGB image (shape: (height, width, 3))

    Returns:
    - gray_image: A 2D numpy array representing the grayscale image (shape: (height, width))
    """
    # Ensure the input is a numpy array
    rgb_image = np.array(rgb_image, dtype=np.float64)
    
    # Extract the R, G, and B channels
    R = rgb_image[:, :, 0]
    G = rgb_image[:, :, 1]
    B = rgb_image[:, :, 2]
    
    # Apply the grayscale conversion formula
    gray_image = (0.299 * R + 0.5870 * G + 0.1140 * B) / 255.0
    gray_image = np.clip(gray_image, 0, 1)
    
    return gray_image


def main():
    image_path = "..\\NeuroSLAM_Datasets\\01_NeuroSLAM_Datasets\\03_QUTCarparkData\\"
    image_source_list = load_image(image_path)
    neuroslam = NeuroSLAM(image_source_list)
    
    gcn = grid_cells_network.GridCellNetwork()
    hdcn = yaw_height_hdc_network.YawHeightHDCNetwork()
    vt = visual_template.VisualTemplateManager()
    vo = visual_odometry.VisualOdometry()
    mep = multilayered_experience_map.ExperienceMap()
    
    gcX, gcY, gcZ = gcn.get_gc_initial_pos()
    curYawTheta, curHeightValue = hdcn.get_hdc_initial_value()
    
    temp, odo_x, odo_y, odo_z = [0.0, 0.0, 0.0, 0.0]
    startFrame = 0
    endFrame = len(image_source_list)
    # endFrame = 1270
    curFrame = 0
    preImg = 0
    n_steps = neuroslam.ODO_STEP
    ########################################################################
    
    odoMap = []
    expMap = []
    t_start = time.time()
    for i in range(startFrame, endFrame, n_steps):
        if i <= len(image_source_list):
            print(f"\nThe {(i // n_steps + 1)} / {endFrame // n_steps} frame is processing ......")
        else:
            break
        curImg = Image.open(load_image(image_path)[i]).convert('L')

        # Visual templates and visual odometry use intensity, so convert to grayscale
        curGrayImg = np.clip(np.uint(curImg), 0, 255).astype(np.uint8)
        curGrayImg = np.float32(curGrayImg / 255.0)

        # Computing the 3D odometry based on the current image
        # yawRotV in degree
        if len(neuroslam.KEY_POINT_SET) == 2:
            if i < neuroslam.KEY_POINT_SET[0]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
            elif i < neuroslam.KEY_POINT_SET[1]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 1)
            else:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 2)
            transV = 2.0
        else:
            if i < neuroslam.KEY_POINT_SET[0]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
            elif i < neuroslam.KEY_POINT_SET[1]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 1)
            elif i < neuroslam.KEY_POINT_SET[2]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
            elif i < neuroslam.KEY_POINT_SET[3]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 2)
            else:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
        
        yawRotV *= neuroslam.DEGREE_TO_RADIAN  # in radian
        
        # Get the most active visual template
        curFrame += 1
        if vt.VT_STEP == 1:
            vtcurGrayImg = np.copy(curGrayImg)
        else:
            if np.mod(curFrame, vt.VT_STEP) == 1:
                vtcurGrayImg = np.copy(curGrayImg)
                preImg = vtcurGrayImg
            else:
                vtcurGrayImg = preImg
        
        vt_id, VT = vt.visual_template(vtcurGrayImg, gcX, gcY, gcZ, curYawTheta, curHeightValue)
        
        # Process the integration of yaw_height_hdc
        [curYawTheta, curHeightValue] = hdcn.yaw_height_hdc_iteration(vt_id, yawRotV, heightV, VT)
        curYawThetaInRadian = curYawTheta * neuroslam.DEGREE_TO_RADIAN  # Transform to radian
        
        # 3D grid cells iteration
        [gcX, gcY, gcZ] = gcn.gc_iteration(vt_id, transV, curYawThetaInRadian, heightV, VT)
        
        # 3D experience map iteration
        mep.exp_map_iteration(vt_id, transV, yawRotV, heightV, gcX, gcY, gcZ, curYawTheta, curHeightValue, VT)
        
        # Update PREV_VT_ID
        vt.PREV_VT_ID = vt_id
        mep.PREV_VT_ID = vt_id
        
        # For drawing visual odometry
        temp += yawRotV
        odo_x += transV * np.cos(temp)  # xcoord
        odo_y += transV * np.sin(temp)  # ycoord
        odo_z += heightV  # zcoord
        odoMap.append([odo_x, odo_y, odo_z])
    
    t_end = time.time()
    print("Time Consumption (s):", t_end - t_start)
    
    for ind in range(mep.NUM_EXPS):
        expTrajectory = [mep.EXPERIENCES[ind].x_exp, mep.EXPERIENCES[ind].y_exp, mep.EXPERIENCES[ind].z_exp]
        expMap.append(expTrajectory)
    
    pass
    
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax1 = fig.add_subplot(121, projection='3d')
    odos = np.array(odoMap)[:, :3]
    np.savetxt("./results/odo_map.txt", odos)
    x1 = odos[:, 0]
    y1 = odos[:, 1]
    z1 = odos[:, 2]
    ax1.scatter(y1 * -0.8, x1 * 0.8, z1 * 0.1)
    plt.title('Odometry')
    
    ax2 = fig.add_subplot(122, projection='3d')
    exps = np.array(expMap)[:, :3]
    np.savetxt("./results/exp_map.txt", exps)
    x2 = exps[:, 0]
    y2 = exps[:, 1]
    z2 = exps[:, 2]
    ax2.scatter(y2 * 0.8, x2 * 0.8, z2 * 0.1)
    plt.title('Experience Map')

    plt.tight_layout()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
