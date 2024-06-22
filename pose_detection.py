"""
Download Pre-Trained Models from: https://github.com/ViTAE-Transformer/ViTPose#animal-datasets-ap10k-apt36k

IMPORTANT: 

If the pose predictions look random/weird, or if you see no pose detections, first split the model using `model_split.sh`
Refer: https://github.com/ViTAE-Transformer/ViTPose/issues/64 - Yes, I spent 1 day figuring this out so you don't have to :)

Author: Animikh Aich
Email: animikh@bu.edu
"""

import cv2
import numpy as np
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result


import torch
import cv2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Constants
video_path = "data/bear.mp4"
pose_checkpoint = "ap10k.pth"
pose_config = 'ViTPose_small_ap10k_256x192.py'
det_thr = 0.5
kpt_thr = 0.3
radius = 4
thickness = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Init Models
pose_model = init_pose_model(
    pose_config, pose_checkpoint, device=str(device)
)

# Init Others
cap = cv2.VideoCapture(video_path)

ctr = 0
old_face_kpts = None

all_face_kpts = list()

# Processing Loop
while cap.isOpened():
    flag, img = cap.read()
    if not flag:
        break

    if img.shape[0] > 800:
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    
    # Object Detection

    boxes = []

    detection_results = []
    for box in boxes:
        x1, y1, x2, y2, confidence, _ = box.tolist()
        if confidence > det_thr:
            val = np.array([x1, y1, x2, y2]).astype(np.int32)
            detection_results.append({"bbox": val})

    # test a single image, with a list of bboxes.
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        img,
        detection_results,
        format="xyxy",
        dataset_info=None,
        return_heatmap=False,
        outputs=None,
    )

    # show the results
    vis_img = vis_pose_result(
        pose_model,
        img,
        pose_results,
        radius=radius,
        thickness=thickness,
        dataset_info=None,
        kpt_score_thr=kpt_thr,
        show=False,
    )

    # # Plot the first 3 keypoints
    # vis_img = img.copy()
    # for pose_result in pose_results:
    #     # Bounding Box
    #     x1, y1, x2, y2 = pose_result["bbox"]
    #     cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #     # Keypoints
    #     face_kpts = list()
    #     for i in range(3):
    #         x, y, score = pose_result["keypoints"][i]
    #         if score > kpt_thr:
    #             cv2.circle(vis_img, (int(x), int(y)), radius, (0, 0, 255), thickness)
            
    #         face_kpts.append([int(x), int(y)])
    
        
    #     if old_face_kpts is not None:
    #         print("Initial Triangle: ", old_face_kpts)
    #         print("New Triangle: ", face_kpts)
    #         rotation_matrix = calculate_plane_orientation_change(old_face_kpts, face_kpts)
    #         plot_triangle_change(old_face_kpts, face_kpts)
        
    #     old_face_kpts = face_kpts


    for pose_result in pose_results:
        face_kpts = list()
        for i in range(3):
            x, y, score = pose_result["keypoints"][i]
            if score > kpt_thr:
                face_kpts.append([int(x), int(y)])

        # Make sure we only record keypoints when all 3 are detected (triangle)
        if len(face_kpts) != 3:
            continue

        all_face_kpts.append(face_kpts)
        

    cv2.imshow("Image", vis_img)
    for pose_result in pose_results:
        # Bounding Box
        x1, y1, x2, y2 = pose_result["bbox"]
        cv2.imshow("Cropped Image", vis_img[y1:y2, x1:x2])

        # Write Image to Disk
        cv2.imwrite(f"../data/images/video1_pose_crop/cropped_image_{ctr}.jpg", vis_img[y1:y2, x1:x2])
        ctr += 1
        
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord(" "):
        cv2.waitKey(0)

# Save the face keypoints as a numpy array

all_face_kpts = np.array(all_face_kpts)
print(all_face_kpts.shape)

# Save the keypoints and the Image Metadata
np.save("../data/other/video1/face_kpts.npy", all_face_kpts)
np.save("../data/other/video1/image_metadata.npy", np.array([vis_img.shape[0], vis_img.shape[1]]))


cap.release()
