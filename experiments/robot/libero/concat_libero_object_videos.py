#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
from typing import List


def find_candidates(root: Path, obj: str) -> List[Path]:
    return sorted(
        p for p in root.rglob("*.mp4")
        if f"pick_up_the_{obj}_and_place_it_in_the_bask" in p.name
    )


def pick_video(candidates: List[Path], prefer_success: bool) -> Path:
    if not candidates:
        raise FileNotFoundError("No matching videos found.")

    if prefer_success:
        success = [p for p in candidates if "--success=True--" in p.name]
        if success:
            return success[0]
    return candidates[0]


def build_ffmpeg_command(
    v1: Path,
    v2: Path,
    v3: Path,
    v4: Path,
    v5: Path,
    v6: Path,
    out: Path,
    obj1: str,
    obj2: str,
    train_a: str,
    train_b: str,
    train_c: str,
) -> List[str]:
    # Layout (2x3):
    # top row    : object1 from C, A, B
    # bottom row : object2 from C, A, B
    label0 = f"{obj1} - Training Scenario"
    label1 = f"Target Object: {obj1} - Change Spawn on Train Distribution {train_a}"
    label2 = f"Target Object: {obj1} - Change Spawn on Train Distribution {train_b}"
    label3 = f"{obj2} - Training Scenario"
    label4 = f"Target Object: {obj2} - Change Spawn on Train Distribution {train_a}"
    label5 = f"Target Object: {obj2} - Change Spawn on Train Distribution {train_b}"

    filter_graph = (
        "[0:v]scale=560:420:force_original_aspect_ratio=decrease,"
        "pad=560:420:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"drawtext=text='{label0}':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5[v0];"
        "[1:v]scale=560:420:force_original_aspect_ratio=decrease,"
        "pad=560:420:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"drawtext=text='{label1}':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5[v1];"
        "[2:v]scale=560:420:force_original_aspect_ratio=decrease,"
        "pad=560:420:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"drawtext=text='{label2}':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5[v2];"
        "[3:v]scale=560:420:force_original_aspect_ratio=decrease,"
        "pad=560:420:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"drawtext=text='{label3}':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5[v3];"
        "[4:v]scale=560:420:force_original_aspect_ratio=decrease,"
        "pad=560:420:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"drawtext=text='{label4}':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5[v4];"
        "[5:v]scale=560:420:force_original_aspect_ratio=decrease,"
        "pad=560:420:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"drawtext=text='{label5}':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5[v5];"
        "[v0][v1][v2]hstack=inputs=3[top];"
        "[v3][v4][v5]hstack=inputs=3[bottom];"
        "[top][bottom]vstack=inputs=2[v]"
    )

    return [
        "ffmpeg", "-y",
        "-i", str(v1),
        "-i", str(v2),
        "-i", str(v3),
        "-i", str(v4),
        "-i", str(v5),
        "-i", str(v6),
        "-filter_complex", filter_graph,
        "-map", "[v]",
        "-an",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        str(out),
    ]


def build_gif_command(mp4_path: Path, gif_path: Path, fps: int = 12, width: int = 960) -> List[str]:
    vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
    return [
        "ffmpeg", "-y",
        "-i", str(mp4_path),
        "-vf", vf,
        str(gif_path),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate 6 LIBERO rollout videos into a 2x3 grid.")
    parser.add_argument(
        "--dir-a",
        default="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/openvla-oft/experiments/robot/libero/rollouts/libero_object/change_spawn_True_train_True",
    )
    parser.add_argument(
        "--dir-b",
        default="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/openvla-oft/experiments/robot/libero/rollouts/libero_object/change_spawn_True_train_False",
    )
    parser.add_argument(
        "--dir-c",
        default="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/openvla-oft/experiments/robot/libero/rollouts/libero_object/change_spawn_False_train_False",
    )
    parser.add_argument(
        "--objects",
        nargs=2,
        default=["alphabet_soup", "cream_cheese"],
    )
    parser.add_argument(
        "--prefer-success",
        action="store_true",
        help="Prefer videos containing '--success=True--' when available.",
    )
    parser.add_argument(
        "--output",
        default="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/openvla-oft/experiments/robot/libero/rollouts/libero_object/concat_alphabet_soup_cream_cheese_trainTrue_vs_trainFalse.mp4",
    )
    args = parser.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    dir_c = Path(args.dir_c)
    obj1, obj2 = args.objects
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    a_obj1 = pick_video(find_candidates(dir_a, obj1), args.prefer_success)
    b_obj1 = pick_video(find_candidates(dir_b, obj1), args.prefer_success)
    c_obj1 = pick_video(find_candidates(dir_c, obj1), args.prefer_success)
    a_obj2 = pick_video(find_candidates(dir_a, obj2), args.prefer_success)
    b_obj2 = pick_video(find_candidates(dir_b, obj2), args.prefer_success)
    c_obj2 = pick_video(find_candidates(dir_c, obj2), args.prefer_success)

    print("Selected videos:")
    print(f"  {obj1} | A: {a_obj1}")
    print(f"  {obj1} | B: {b_obj1}")
    print(f"  {obj1} | C: {c_obj1}")
    print(f"  {obj2} | A: {a_obj2}")
    print(f"  {obj2} | B: {b_obj2}")
    print(f"  {obj2} | C: {c_obj2}")

    cmd = build_ffmpeg_command(
        c_obj1,
        a_obj1,
        b_obj1,
        c_obj2,
        a_obj2,
        b_obj2,
        out,
        obj1=obj1,
        obj2=obj2,
        train_a="True",
        train_b="False",
        train_c="False",
    )
    subprocess.run(cmd, check=True)
    print(f"\nCreated MP4: {out}")

    gif_out = out.with_suffix(".gif")
    gif_cmd = build_gif_command(out, gif_out)
    subprocess.run(gif_cmd, check=True)
    print(f"Created GIF: {gif_out}")


if __name__ == "__main__":
    main()
