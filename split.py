import os
import shutil
from oord_dataset import InferDataset
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style


def list_sequences(dataset_path):
    """
    List available sequence folders in dataset_path/cartesian or dataset_path/stacked.
    Assumes each sequence has a folder with suffix _resize.
    """
    cartesian_path = os.path.join(dataset_path, 'cartesian')
    stacked_path = os.path.join(dataset_path, 'stacked')
    sequences = set()

    for path in [cartesian_path, stacked_path]:
        if os.path.exists(path):
            for entry in os.listdir(path):
                if entry.endswith('_resize'):
                    seq = entry.replace('_resize', '')
                    sequences.add(seq)

    return sorted(list(sequences))


def extract_images_in_interval(sequence, dataset_path, start_ts, end_ts, output_dir, name):
    dataset = InferDataset(sequence, dataset_path=dataset_path)
    dest = os.path.join(output_dir, sequence + '_' + name)
    os.makedirs(dest, exist_ok=True)

    num_images = 0
    for i, ts in enumerate(dataset.timestamps):
        if start_ts <= ts <= end_ts:
            num_images += 1
            src_img_path = dataset.imgs_path[i]
            dst_img_path = os.path.join(dest, os.path.basename(src_img_path))
            shutil.copy(src_img_path, dst_img_path)

    print(f"✅ Saved {num_images} radar images between {start_ts} and {end_ts} to: {dest}")


def main():
    # === Style for prompt ===
    style = Style.from_dict({'': '#00ff00'})

    # === Dataset path and type ===
    dataset_path = input("📁 Dataset path (default: ../../oord_data/): ").strip() or "../../oord_data/"
    output_path = input("📂 Output folder (default: ../../oord_data/test_seq): ").strip() or "../../oord_data/test_seq"
    data_type = input("📦 Data type (cartesian/stacked, default: cartesian): ").strip() or "cartesian"
    name = input("🏷️ Name suffix (optional): ").strip()

    # === Autocomplete for sequence ===
    sequences = list_sequences(dataset_path)
    if not sequences:
        print("❌ No sequences found in dataset. Check your dataset path.")
        return

    sequence_completer = WordCompleter(sequences, ignore_case=True)
    sequence = prompt("📌 Sequence name: ", completer=sequence_completer, style=style).strip()
    if sequence not in sequences:
        print(f"❌ Sequence '{sequence}' not found.")
        return

    # === Load dataset to fetch timestamps ===
    dataset = InferDataset(sequence, dataset_path=dataset_path)
    timestamps = [str(ts) for ts in dataset.timestamps]
    timestamp_completer = WordCompleter(timestamps, ignore_case=True)

    print("\nStart typing timestamp and press TAB to autocomplete:\n")
    start_ts = int(prompt("⏱️ Begin timestamp: ", completer=timestamp_completer, style=style))
    end_ts = int(prompt("⏱️ End timestamp:   ", completer=timestamp_completer, style=style))

    # === Run extraction ===
    extract_images_in_interval(sequence, dataset_path, start_ts, end_ts, output_path, name)


if __name__ == "__main__":
    main()
