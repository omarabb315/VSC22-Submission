"""
Convert VCDB core dataset (or arbitrary video folders) into the directory layout
and metadata format expected by the VSC22 pipeline.

Usage:
    # VCDB mode:
    python prepare_data.py --vcdb_dir vcdb_core/ [--max_videos 6]

    # Generic mode (any two folders of MP4s):
    python prepare_data.py --query_dir /path/to/queries --ref_dir /path/to/references
"""

import argparse
import csv
import json
import os
import glob
import shutil
import subprocess
import zipfile
from pathlib import Path
from multiprocessing import Pool, cpu_count


DESCRIPTOR_ROOT = Path(__file__).parent / "VSC22-Descriptor-Track-1st"
DATA_ROOT = DESCRIPTOR_ROOT / "data"


def ffprobe_video(video_path):
    """Probe video for duration, fps, width, height using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)
        stream = info.get("streams", [{}])[0]
        fmt = info.get("format", {})

        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))

        r_frame_rate = stream.get("r_frame_rate", "30/1")
        if "/" in r_frame_rate:
            num, den = r_frame_rate.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(r_frame_rate) if r_frame_rate else 30.0

        duration = float(stream.get("duration", 0) or fmt.get("duration", 0) or 0)

        return {"duration": duration, "fps": round(fps, 2), "width": width, "height": height}
    except Exception as e:
        print(f"WARN: ffprobe failed for {video_path}: {e}")
        return {"duration": 0, "fps": 30.0, "width": 0, "height": 0}


def extract_frames_to_zip(args):
    """Extract frames at 1fps from a video into a zip file."""
    video_path, zip_path = args
    if os.path.isfile(zip_path):
        return 0

    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    tmp_dir = zip_path + "_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = f'ffmpeg -nostdin -y -i "{video_path}" -start_number 0 -q 0 -vf fps=1 "{tmp_dir}/%07d.jpg"'
    try:
        subprocess.run(cmd, shell=True, timeout=120, capture_output=True)
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return -1

    imgpaths = sorted(glob.glob(os.path.join(tmp_dir, "*.jpg")))
    if imgpaths:
        with zipfile.ZipFile(zip_path, "w") as wzip:
            for p in imgpaths:
                wzip.write(p, arcname=os.path.basename(p))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


def discover_vcdb_videos(vcdb_dir):
    """
    Discover VCDB core dataset structure.
    VCDB is organized as: vcdb_dir/<query_group>/<video_files>
    Each query group folder has related videos (originals + copies).
    Returns list of (group_name, video_path) tuples.
    """
    vcdb_dir = Path(vcdb_dir)
    videos = []
    video_exts = {".mp4", ".avi", ".flv", ".mkv", ".webm", ".mov", ".wmv"}

    for group_dir in sorted(vcdb_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name
        for vf in sorted(group_dir.iterdir()):
            if vf.suffix.lower() in video_exts:
                videos.append((group_name, str(vf)))

    return videos


def discover_vcdb_annotations(vcdb_dir):
    """
    Parse VCDB annotation files.
    Annotation format (one per query group): pairs of copied segments.
    """
    vcdb_dir = Path(vcdb_dir)
    annotations = []

    for ann_file in sorted(vcdb_dir.rglob("*.txt")):
        try:
            with open(ann_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        annotations.append({
                            "video_a": parts[0],
                            "start_a": float(parts[1]),
                            "end_a": float(parts[2]),
                            "video_b": parts[3],
                            "start_b": float(parts[4]),
                            "end_b": float(parts[5]),
                        })
        except Exception as e:
            print(f"WARN: Could not parse annotation {ann_file}: {e}")

    return annotations


def split_vcdb_to_query_ref(vcdb_videos, max_videos=None):
    """
    Split VCDB videos into query and reference sets.
    Strategy: for each group, the first video is the 'reference',
    the rest are 'queries' (potential copies).
    """
    groups = {}
    for group_name, video_path in vcdb_videos:
        groups.setdefault(group_name, []).append(video_path)

    ref_videos = []
    query_videos = []

    for group_name in sorted(groups.keys()):
        vids = groups[group_name]
        ref_videos.append(vids[0])
        query_videos.extend(vids[1:])

    if max_videos:
        half = max(max_videos // 2, 1)
        ref_videos = ref_videos[:half]
        query_videos = query_videos[:half]

    return query_videos, ref_videos


def setup_data_directories():
    """Create the required directory structure."""
    dirs = [
        DATA_ROOT / "videos" / "test" / "query",
        DATA_ROOT / "videos" / "test" / "reference",
        DATA_ROOT / "videos" / "train" / "query",
        DATA_ROOT / "videos" / "train" / "reference",
        DATA_ROOT / "meta" / "test",
        DATA_ROOT / "meta" / "train",
        DATA_ROOT / "meta" / "val",
        DATA_ROOT / "jpg_zips",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def create_symlinks_and_metadata(query_videos, ref_videos):
    """
    Create symlinks with standardized names and generate metadata files.
    Returns (id_mapping, query_meta_rows, ref_meta_rows).
    """
    id_mapping = {}
    query_meta_rows = []
    ref_meta_rows = []

    for rn, video_path in enumerate(ref_videos, start=1):
        vid_id = f"R{rn:06d}"
        ext = Path(video_path).suffix
        link_path = DATA_ROOT / "videos" / "test" / "reference" / f"{vid_id}{ext}"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(Path(video_path).resolve())
        id_mapping[vid_id] = str(video_path)

        info = ffprobe_video(video_path)
        ref_meta_rows.append({
            "video_id": vid_id, "duration_sec": info["duration"],
            "frames_per_sec": info["fps"], "width": info["width"],
            "height": info["height"], "rn": rn,
        })

    for rn, video_path in enumerate(query_videos, start=1):
        vid_id = f"Q{rn:06d}"
        ext = Path(video_path).suffix
        link_path = DATA_ROOT / "videos" / "test" / "query" / f"{vid_id}{ext}"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(Path(video_path).resolve())
        id_mapping[vid_id] = str(video_path)

        info = ffprobe_video(video_path)
        query_meta_rows.append({
            "video_id": vid_id, "duration_sec": info["duration"],
            "frames_per_sec": info["fps"], "width": info["width"],
            "height": info["height"], "rn": rn,
        })

    return id_mapping, query_meta_rows, ref_meta_rows


def write_metadata_files(query_meta_rows, ref_meta_rows, id_mapping):
    """Write all metadata CSVs and vid list files."""
    fields = ["video_id", "duration_sec", "frames_per_sec", "width", "height", "rn"]

    with open(DATA_ROOT / "meta" / "test" / "test_query_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(query_meta_rows)

    with open(DATA_ROOT / "meta" / "test" / "test_reference_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ref_meta_rows)

    ref_ids = [r["video_id"] for r in ref_meta_rows]
    query_ids = [r["video_id"] for r in query_meta_rows]

    with open(DATA_ROOT / "meta" / "test" / "test_ref_vids.txt", "w") as f:
        f.write("\n".join(ref_ids) + "\n")

    # Train refs = same as test refs (needed for score normalization)
    with open(DATA_ROOT / "meta" / "train" / "train_ref_vids.txt", "w") as f:
        f.write("\n".join(ref_ids) + "\n")

    # Train reference metadata = copy of test reference metadata
    with open(DATA_ROOT / "meta" / "train" / "train_reference_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ref_meta_rows)

    # Train query metadata = copy of test query metadata
    with open(DATA_ROOT / "meta" / "train" / "train_query_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(query_meta_rows)

    with open(DATA_ROOT / "meta" / "train" / "train_query_vids.txt", "w") as f:
        f.write("\n".join(query_ids) + "\n")

    # Vids.txt = all video IDs
    all_ids = ref_ids + query_ids
    with open(DATA_ROOT / "meta" / "vids.txt", "w") as f:
        f.write("\n".join(all_ids) + "\n")

    # Train vids = all
    with open(DATA_ROOT / "meta" / "train" / "train_vids.txt", "w") as f:
        f.write("\n".join(all_ids) + "\n")

    # Val metadata (empty but required)
    with open(DATA_ROOT / "meta" / "val" / "val_query_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    # Save ID mapping
    mapping_path = DESCRIPTOR_ROOT / "id_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(id_mapping, f, indent=2)
    print(f"ID mapping saved to {mapping_path}")


def extract_all_frames(id_mapping):
    """Extract 1fps frames for all videos into jpg_zips/."""
    tasks = []
    for vid_id, original_path in id_mapping.items():
        suffix = vid_id[-2:]
        zip_path = str(DATA_ROOT / "jpg_zips" / suffix / f"{vid_id}.zip")
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        ext = Path(original_path).suffix
        video_path = str(DATA_ROOT / "videos" / "test" /
                        ("query" if vid_id.startswith("Q") else "reference") /
                        f"{vid_id}{ext}")
        tasks.append((video_path, zip_path))

    num_workers = min(cpu_count(), 8, len(tasks))
    print(f"Extracting frames for {len(tasks)} videos using {num_workers} workers...")

    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = pool.map(extract_frames_to_zip, tasks)
    else:
        results = [extract_frames_to_zip(t) for t in tasks]

    failed = sum(1 for r in results if r != 0)
    if failed:
        print(f"WARN: {failed}/{len(tasks)} frame extractions failed")
    else:
        print(f"All {len(tasks)} videos processed successfully")


def main():
    parser = argparse.ArgumentParser(description="Prepare video data for VSC22 pipeline")
    parser.add_argument("--vcdb_dir", type=str, default=None,
                       help="Path to VCDB core dataset root directory")
    parser.add_argument("--query_dir", type=str, default=None,
                       help="Path to directory containing query videos (generic mode)")
    parser.add_argument("--ref_dir", type=str, default=None,
                       help="Path to directory containing reference videos (generic mode)")
    parser.add_argument("--max_videos", type=int, default=None,
                       help="Limit total videos for testing (splits evenly between query/ref)")
    args = parser.parse_args()

    if not args.vcdb_dir and not (args.query_dir and args.ref_dir):
        parser.error("Provide either --vcdb_dir or both --query_dir and --ref_dir")

    setup_data_directories()

    if args.vcdb_dir:
        print(f"=== VCDB mode: scanning {args.vcdb_dir} ===")
        vcdb_videos = discover_vcdb_videos(args.vcdb_dir)
        print(f"Found {len(vcdb_videos)} videos in {len(set(g for g, _ in vcdb_videos))} groups")
        query_videos, ref_videos = split_vcdb_to_query_ref(vcdb_videos, args.max_videos)
    else:
        print(f"=== Generic mode: query={args.query_dir}, ref={args.ref_dir} ===")
        video_exts = {".mp4", ".avi", ".flv", ".mkv", ".webm", ".mov", ".wmv"}
        query_videos = sorted([
            str(p) for p in Path(args.query_dir).iterdir()
            if p.suffix.lower() in video_exts
        ])
        ref_videos = sorted([
            str(p) for p in Path(args.ref_dir).iterdir()
            if p.suffix.lower() in video_exts
        ])
        if args.max_videos:
            half = max(args.max_videos // 2, 1)
            query_videos = query_videos[:half]
            ref_videos = ref_videos[:half]

    print(f"Query videos: {len(query_videos)}, Reference videos: {len(ref_videos)}")

    print("Creating symlinks and probing video metadata...")
    id_mapping, query_meta, ref_meta = create_symlinks_and_metadata(query_videos, ref_videos)

    print("Writing metadata files...")
    write_metadata_files(query_meta, ref_meta, id_mapping)

    print("Extracting frames at 1fps...")
    extract_all_frames(id_mapping)

    print(f"\n=== Data preparation complete ===")
    print(f"  Queries: {len(query_videos)}")
    print(f"  References: {len(ref_videos)}")
    print(f"  Data root: {DATA_ROOT}")


if __name__ == "__main__":
    main()
