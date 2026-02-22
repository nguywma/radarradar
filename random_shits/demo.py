import os
import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from oord_dataset import InferDataset
def save_one_pair_custom_output(query_seq, db_seq, dataset_path='../../oord_data/',
                                save_path='./output', close_thresh=25, far_thresh=50):
    # Load datasets
    query_dataset = InferDataset(query_seq, dataset_path)
    db_dataset = InferDataset(db_seq, dataset_path)

    # Get GPS positions
    query_pos_dict = InferDataset.get_radar_positions(query_dataset.poses, query_dataset.timestamps)
    db_pos_dict = InferDataset.get_radar_positions(db_dataset.poses, db_dataset.timestamps)

    # Prepare save folders
    query_out_folder = os.path.join(save_path, query_seq)
    db_out_folder = os.path.join(save_path, db_seq)
    os.makedirs(query_out_folder, exist_ok=True)
    os.makedirs(db_out_folder, exist_ok=True)

    for i, query_ts in enumerate(query_dataset.timestamps):
        query_img_tensor, _ = query_dataset[i]
        query_pos = query_pos_dict[query_ts]

        close_idx = None
        far_idx = None
        min_close_dist = float('inf')
        max_far_dist = 0

        for j, db_ts in enumerate(db_dataset.timestamps):
            db_pos = db_pos_dict[db_ts]
            dist = euclidean(query_pos, db_pos)

            if dist <= close_thresh and dist < min_close_dist:
                min_close_dist = dist
                close_idx = j

            if dist >= far_thresh and dist > max_far_dist:
                max_far_dist = dist
                far_idx = j

        if close_idx is not None and far_idx is not None:
            # Convert images from tensor to grayscale
            query_img = (query_img_tensor[0] * 256).astype(np.uint8)
            close_img = (db_dataset[close_idx][0][0] * 256).astype(np.uint8)
            far_img = (db_dataset[far_idx][0][0] * 256).astype(np.uint8)

            # Get original names
            query_name = os.path.basename(query_dataset.imgs_path[i])
            close_name = os.path.basename(db_dataset.imgs_path[close_idx])
            far_name = os.path.basename(db_dataset.imgs_path[far_idx])

            # New save paths
            cv2.imwrite(os.path.join(query_out_folder, query_name.replace('.png', '.png')), query_img)
            cv2.imwrite(os.path.join(db_out_folder, f'{close_name}'), close_img)
            cv2.imwrite(os.path.join(db_out_folder, f'{far_name}'), far_img)

            print(f"✅ Saved to: {query_out_folder}, {db_out_folder}")
            print(f"    - query: {query_name}")
            print(f"    - db_close: {close_name} ({min_close_dist:.1f}m)")
            print(f"    - db_far:   {far_name} ({max_far_dist:.1f}m)")
            return  # only save one pair

        else:
            print(f"❌ Skipping query index {i}: no valid close/far match")

    print("⚠️ No valid query-db pair found in sequences.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--query_seq', required=True, help='Query sequence name (e.g., Bellmouth_1)')
    parser.add_argument('--db_seq', required=True, help='Database sequence name (e.g., Bellmouth_2)')
    parser.add_argument('--dataset_path', default='../../oord_data/', help='Dataset base path')
    parser.add_argument('--save_path', default='./output', help='Where to save query/db folders')
    parser.add_argument('--close_thresh', type=float, default=25, help='Close match threshold (meters)')
    parser.add_argument('--far_thresh', type=float, default=50, help='Far match threshold (meters)')

    args = parser.parse_args()

    save_one_pair_custom_output(
        args.query_seq,
        args.db_seq,
        args.dataset_path,
        args.save_path,
        args.close_thresh,
        args.far_thresh
    )
