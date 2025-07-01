from PIL import Image
import numpy as np
import mpl_toolkits.mplot3d
import os
import matplotlib.pyplot as plt
import visual_odometry
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


def main():
    image_path = "..\\NeuroSLAM_Datasets\\01_NeuroSLAM_Datasets\\03_QUTCarparkData\\"
    # image_path = "..\\NeuroSLAM_Datasets\\01_NeuroSLAM_Datasets\\04_vo_data\\"
    image_source_list = load_image(image_path)
    neuroslam = NeuroSLAM(image_source_list)
    vo = visual_odometry.VisualOdometry()
    
    temp, odo_x, odo_y, odo_z = [0.0, 0.0, 0.0, 0.0]
    startFrame = 0
    endFrame = len(image_source_list)
    # endFrame = 4800
    n_steps = neuroslam.ODO_STEP
    ########################################################################
    
    odoMap = []
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
        if len(neuroslam.KEY_POINT_SET) == 8:
            if i < neuroslam.KEY_POINT_SET[0]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
            elif i < neuroslam.KEY_POINT_SET[1]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 1)
            elif i < neuroslam.KEY_POINT_SET[2]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
            elif i < neuroslam.KEY_POINT_SET[3]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 2)
            elif i < neuroslam.KEY_POINT_SET[4]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
            elif i < neuroslam.KEY_POINT_SET[5]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 1)
            elif i < neuroslam.KEY_POINT_SET[6]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
            elif i < neuroslam.KEY_POINT_SET[7]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 2)
            else:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
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
        
        # For drawing visual odometry
        temp += yawRotV
        odo_x += transV * np.cos(temp)  # xcoord
        odo_y += transV * np.sin(temp)  # ycoord
        odo_z += heightV  # zcoord
        odoMap.append([odo_x, odo_y, odo_z])
    
    t_end = time.time()
    print("Time Consumption (s):", t_end - t_start)
    
    pass
    
    fig = plt.figure(figsize=(5, 7), dpi=100)
    ax1 = fig.add_subplot(111, projection='3d')
    odos = np.array(odoMap)[:, :3]
    x1 = odos[:, 0]
    y1 = odos[:, 1]
    z1 = odos[:, 2]
    ax1.scatter(y1 * 0.8, x1 * 0.8, z1 * 0.1)
    plt.title('Visual Odometry')

    plt.tight_layout()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
