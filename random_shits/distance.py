import numpy as np
from oord_dataset import InferDataset
def average_scan_distance(dataset: InferDataset, min_dist_threshold=0.1):
    """
    Compute the average distance between consecutive scans for a dataset
    using InferDataset.get_radar_positions(), filtering out near-zero moves.
    
    Args:
        dataset (InferDataset): the dataset instance
        min_dist_threshold (float): minimum distance (in meters) to count
    """
    # Map image timestamps to GPS positions
    radar_positions = dataset.get_radar_positions(dataset.poses, dataset.timestamps)

    # Extract positions in timestamp order
    positions = [radar_positions[t] for t in dataset.timestamps]

    # Compute distances
    distances = [
        np.linalg.norm(positions[i] - positions[i-1])
        for i in range(1, len(positions))
    ]

    # Filter out near-zero distances
    filtered = [d for d in distances if d > min_dist_threshold]

    return float(np.mean(filtered)) if filtered else 0.0


if __name__ == "__main__":
    seqs = ["Bellmouth_1", "Bellmouth_2", "Bellmouth_3", "Bellmouth_4","Maree_1", "Maree_2", "Hydro_2", "Hydro_3", "Twolochs_1", "Twolochs_2", "Hydro_1"]  # add more sequences
    for seq in seqs:
        dataset = InferDataset(seq, dataset_path='../../oord_data/', sample_inteval=1)
        avg_dist = average_scan_distance(dataset, min_dist_threshold=1)
        print(f"{seq}: average scan distance = {avg_dist:.3f} meters")
