import numpy as np
import mpl_toolkits.mplot3d
import os
import matplotlib.pyplot as plt
import pandas as pd


def RTK():
    RTK_Data = pd.read_excel(r'C:\Users\HX\Desktop\rtk.xlsx', usecols=[2, 3, 4])
    RTK_Data = np.array(RTK_Data)
    
    Anchors = RTK_Data[::-1]
    Anchors -= Anchors[0]
    
    return Anchors


plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(RTK()[:107, 0], RTK()[:107, 1], RTK()[:107, 2], color='red')
ax.plot(RTK()[:157, 0], RTK()[:157, 1], RTK()[:157, 2], color='red')
plt.tight_layout()
plt.show()
