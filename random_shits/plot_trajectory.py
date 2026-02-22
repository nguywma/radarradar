import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from oord_dataset import InferDataset

# === Configuration ===
dataset_path = '../../oord_data/'
all_sequences = [
    # 'Bellmouth_1', 'Bellmouth_2', #'Bellmouth_3', 'Bellmouth_4',
    # 'Hydro_1', 'Hydro_2', 'Hydro_3',
    # 'Maree_1', 'Maree_2',
    'Twolochs_1', 'Twolochs_2'
]
group_prefixes = ['Bellmouth', 'Hydro', 'Maree', 'Twolochs']
sample_interval = 1
frame_interval_ms = 30
speed_up = 5
output_dir = '../../oord_data/test_seq/animations_by_group'
os.makedirs(output_dir, exist_ok=True)

# === Group sequences by prefix ===
grouped_sequences = {prefix: [] for prefix in group_prefixes}
for seq in all_sequences:
    for prefix in group_prefixes:
        if seq.startswith(prefix):
            grouped_sequences[prefix].append(seq)
            break

# === Animate each group ===
for group_name, sequences in grouped_sequences.items():
    print(f"\n=== Animating group: {group_name} ===")
    output_file = os.path.join(output_dir, f'{group_name}.mp4')

    all_trajs = []
    for seq in sequences:
        dataset = InferDataset(seq, dataset_path=dataset_path, sample_inteval=sample_interval)
        pos_dict = InferDataset.get_radar_positions(dataset.poses, dataset.timestamps)

        northings, eastings, timestamps = [], [], []
        for ts in dataset.timestamps:
            pos = pos_dict.get(ts)
            if pos is not None:
                northings.append(pos[0])
                eastings.append(pos[1])
                timestamps.append(ts)

        if len(northings) > 1:
            distances = [0.0]
            speeds = [0.0]
            for i in range(1, len(northings)):
                dx = eastings[i] - eastings[i - 1]
                dy = northings[i] - northings[i - 1]
                dist = np.hypot(dx, dy)

                dt = timestamps[i] - timestamps[i - 1]  # seconds (UNIX time)
                if dt > 1e-6:
                    speed = dist / dt
                else:
                    speed = 0.0

                distances.append(distances[-1] + dist)
                speeds.append(speed)

            all_trajs.append({
                'seq': seq,
                'eastings': eastings,
                'northings': northings,
                'timestamps': timestamps,
                'distances': distances,
                'speeds': speeds
            })
        else:
            print(f"⚠️ Skipping {seq} (insufficient data)")

    if len(all_trajs) == 0:
        print(f"❌ No data to animate for {group_name}. Skipping.")
        continue

    # Build frame map
    frame_map = []
    for i, traj in enumerate(all_trajs):
        for f in range(0, len(traj['eastings']), speed_up):
            frame_map.append((i, f))

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 8))
    lines = [ax.plot([], [], lw=2, alpha=0.2)[0] for _ in all_trajs]
    point, = ax.plot([], [], 'ro', markersize=5)
    title = ax.text(0.5, 1.01, '', transform=ax.transAxes, ha='center', fontsize=12)
    timestamp_text = ax.text(0.01, 0.99, '', transform=ax.transAxes,
                             ha='left', va='top', fontsize=10, color='blue')
    distance_text = ax.text(0.01, 0.94, '', transform=ax.transAxes,
                            ha='left', va='top', fontsize=10, color='green')
    speed_text = ax.text(0.01, 0.89, '', transform=ax.transAxes,
                         ha='left', va='top', fontsize=10, color='red')

    all_eastings = np.concatenate([np.array(t['eastings']) for t in all_trajs])
    all_northings = np.concatenate([np.array(t['northings']) for t in all_trajs])
    margin = 10
    ax.set_xlim(all_eastings.min() - margin, all_eastings.max() + margin)
    ax.set_ylim(all_northings.min() - margin, all_northings.max() + margin)
    ax.set_xlabel('UTM Easting')
    ax.set_ylabel('UTM Northing')
    ax.set_title(f'Trajectories: {group_name}')
    ax.set_aspect('equal')
    ax.grid(True)

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_alpha(0.2)
        point.set_data([], [])
        title.set_text('')
        timestamp_text.set_text('')
        distance_text.set_text('')
        speed_text.set_text('')
        return lines + [point, title, timestamp_text, distance_text, speed_text]

    def update(global_frame):
        seq_idx, frame_idx = frame_map[global_frame]
        traj = all_trajs[seq_idx]

        for idx, line in enumerate(lines):
            t = all_trajs[idx]
            if idx < seq_idx:
                line.set_data(t['eastings'], t['northings'])
                line.set_alpha(0.1)
            elif idx == seq_idx:
                line.set_data(t['eastings'][:frame_idx + 1], t['northings'][:frame_idx + 1])
                line.set_alpha(0.9)
                point.set_data([t['eastings'][frame_idx]], [t['northings'][frame_idx]])
            else:
                line.set_data([], [])
                line.set_alpha(0.2)

        title.set_text(f'Sequence: {traj["seq"]}')
        ts = traj['timestamps'][frame_idx]
        dist = traj['distances'][frame_idx]
        speed = traj['speeds'][frame_idx]

        timestamp_text.set_text(f'Timestamp: {ts:.3f}')
        distance_text.set_text(f'Distance: {dist:.2f} m')
        speed_text.set_text(f'Speed: {speed:.2f} m/s')

        return lines + [point, title, timestamp_text, distance_text, speed_text]

    print("Rendering animation...")
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=tqdm(range(len(frame_map)), desc=f"Rendering {group_name}"),
        init_func=init,
        blit=True,
        interval=frame_interval_ms
    )

    print(f"Saving to {output_file}...")
    ani.save(output_file, writer='ffmpeg', dpi=150)
    plt.close()
    print(f"✅ Saved {output_file}")
