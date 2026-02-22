# import os
# import cv2
# import numpy as np
# from oord_dataset import InferDataset


# def get_angle_from_matrix(R):
#     """Extract yaw angle in degrees from the SE(2) rotation matrix."""
#     return np.degrees(np.arctan2(R[1, 0], R[0, 0]))


# def save_image(img_tensor, path):
#     img = (img_tensor[0] * 256).astype(np.uint8)
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     cv2.imwrite(path, img)


# # 1. Initialize Datasets
# ds1 = InferDataset(seq='Hydro_1', sample_inteval=1)
# ds2 = InferDataset(seq='Hydro_2', sample_inteval=1)

# # 2. Reference Image (Hydro_1)
# idx1 = 6500
# ref_ts = ds1.timestamps[idx1]
# ref_utm = InferDataset.get_radar_positions(ds1.poses, [ref_ts])[ref_ts]
# ref_yaw_mat = InferDataset.get_yaw(ds1.imu, ref_ts, ds1.poses)
# ref_angle = get_angle_from_matrix(ref_yaw_mat)

# # 3. Search for a Positive Sample with different orientation in Hydro_2
# pos2 = InferDataset.get_radar_positions(ds2.poses, ds2.timestamps)

# found_idx = None
# orientation_diff = None
# matched_dist = None
# target_angle = None

# for i, ts in enumerate(ds2.timestamps):
#     dist = np.linalg.norm(pos2[ts] - ref_utm)

#     # Positive defined as spatially close
#     if dist < 20:
#         target_yaw_mat = InferDataset.get_yaw(ds2.imu, ts, ds2.poses)
#         angle = get_angle_from_matrix(target_yaw_mat)

#         # Shortest angular distance
#         diff = abs((angle - ref_angle + 180) % 360 - 180)

#         # Large heading change
#         if diff > 80:
#             found_idx = i
#             orientation_diff = diff
#             matched_dist = dist
#             target_angle = angle
#             break

# # 4. Save and Report Results
# if found_idx is not None:
#     ref_img, _ = ds1[idx1]
#     pos_img, _ = ds2[found_idx]

#     name1 = os.path.basename(ds1.imgs_path[idx1])
#     name2 = os.path.basename(ds2.imgs_path[found_idx])

#     save_image(
#         ref_img,
#         f'/media/manh/manh/oord_data/test_seq/test_4_paper/cartesian/Hydro_1_resize/{name1}'
#     )
#     save_image(
#         pos_img,
#         f'/media/manh/manh/oord_data/test_seq/test_4_paper/cartesian/Hydro_2_resize/{name2}'
#     )

#     print("Match Found!")
#     print(f"Distance Difference: {matched_dist:.2f} m")
#     print(f"Reference Angle: {ref_angle:.2f}°")
#     print(f"Positive Sample Angle: {target_angle:.2f}°")
#     print(f"Orientation Difference: {orientation_diff:.2f}°")

# else:
#     print("No positive samples with a significant orientation difference were found.")
import os
import cv2
import numpy as np
from oord_dataset import InferDataset


def get_angle_from_matrix(R):
    return np.degrees(np.arctan2(R[1, 0], R[0, 0]))


def save_image(img_tensor, path):
    img = (img_tensor[0] * 256).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


# -------------------- Parameters --------------------
POS_DIST = 5.0
NEG_DIST = 25.0
POS_ORI_THRESH = 80.0
NUM_NEG = 10

# -------------------- Datasets --------------------
ds1 = InferDataset(seq='Bellmouth_1', sample_inteval=1)
ds2 = InferDataset(seq='Bellmouth_2', sample_inteval=1)

# -------------------- Reference --------------------
idx1 = 2000
ref_ts = ds1.timestamps[idx1]
ref_utm = InferDataset.get_radar_positions(ds1.poses, [ref_ts])[ref_ts]
ref_yaw = get_angle_from_matrix(
    InferDataset.get_yaw(ds1.imu, ref_ts, ds1.poses)
)

# -------------------- Search --------------------
pos2 = InferDataset.get_radar_positions(ds2.poses, ds2.timestamps)

positive = None
negatives = []

for i, ts in enumerate(ds2.timestamps):
    dist = np.linalg.norm(pos2[ts] - ref_utm)

    yaw = get_angle_from_matrix(
        InferDataset.get_yaw(ds2.imu, ts, ds2.poses)
    )
    diff = abs((yaw - ref_yaw + 180) % 360 - 180)

    # -------- Positive: very close + large rotation --------
    if positive is None and dist <= POS_DIST and diff > POS_ORI_THRESH:
        positive = {
            "idx": i,
            "dist": dist,
            "diff": diff,
            "yaw": yaw
        }

    # -------- Negative: far (≥ 25m) --------
    elif dist >= NEG_DIST:
        negatives.append({
            "idx": i,
            "dist": dist,
            "diff": diff,
            "yaw": yaw
        })

# -------------------- Pick hardest negatives --------------------
if positive is not None and len(negatives) >= NUM_NEG:
    negatives = sorted(
        negatives,
        key=lambda x: abs(x["dist"] - NEG_DIST)
    )[:NUM_NEG]

    # -------------------- Save reference --------------------
    ref_img, _ = ds1[idx1]
    ref_name = os.path.basename(ds1.imgs_path[idx1])
    save_image(
        ref_img,
        f'/media/manh/manh/oord_data/test_seq/test_4_paper/cartesian/Bellmouth_1_resize/{ref_name}'
    )

    # -------------------- Save positive --------------------
    pos_img, _ = ds2[positive["idx"]]
    pos_name = os.path.basename(ds2.imgs_path[positive["idx"]])
    save_image(
        pos_img,
        f'/media/manh/manh/oord_data/test_seq/test_4_paper/cartesian/Bellmouth_2_resize/pos_{pos_name}'
    )

    # -------------------- Save negatives --------------------
    for k, neg in enumerate(negatives):
        neg_img, _ = ds2[neg["idx"]]
        neg_name = os.path.basename(ds2.imgs_path[neg["idx"]])
        save_image(
            neg_img,
            f'/media/manh/manh/oord_data/test_seq/test_4_paper/cartesian/Bellmouth_2_resize/{neg_name}'
        )

    # -------------------- Report --------------------
    print("Positive sample:")
    print(f"  Distance: {positive['dist']:.2f} m")
    print(f"  Orientation diff: {positive['diff']:.2f}°")

    print("\nNegative samples:")
    for k, neg in enumerate(negatives):
        print(
            f"  {k}: dist={neg['dist']:.2f} m, "
            f"angle diff={neg['diff']:.2f}°"
        )

else:
    print("No valid positive or not enough negatives found.")
