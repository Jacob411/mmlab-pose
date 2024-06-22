import cv2
import sys
import matplotlib.pyplot as plt
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
import torch
# Use command line argument as the image path
img = sys.argv[1]
pose_config = 'configs/ViTPose_small_ap10k_256x192.py'
pose_checkpoint = 'weights/ap10k.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize pose model
pose_model = init_pose_model(pose_config, pose_checkpoint, device=str(device))




# inference pose
pose_results, returned_outputs = inference_top_down_pose_model(pose_model,
                                                               img,
                                                               format='xyxy',
                                                               dataset=pose_model.cfg.data.test.type)

# show pose estimation results
vis_result = vis_pose_result(pose_model,
                             img,
                             pose_results,
                             dataset=pose_model.cfg.data.test.type)
# reduce image size
print(pose_results)

plt.imshow(cv2.cvtColor(vis_result, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Turn off axis labels
plt.show()

