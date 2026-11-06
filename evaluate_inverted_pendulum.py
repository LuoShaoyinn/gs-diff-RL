"""Evaluate a saved Genesis-trained policy on the unmodified Gymnasium benchmark."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch

from train_inverted_pendulum import Policy, evaluate_gym, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", default="runs/inverted-pendulum-evaluation")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = Policy()
    policy.load_state_dict(checkpoint["policy"])
    result = evaluate_gym(policy, args.seed, args.episodes, out / "trace.csv")
    result["checkpoint"] = str(Path(args.checkpoint).resolve())
    result["iteration"] = checkpoint["iteration"]
    write_json(out / "evaluation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    if args.video:
        import av

        env = gym.make("InvertedPendulum-v5", render_mode="rgb_array", width=640, height=480)
        obs, _ = env.reset(seed=args.seed)
        container = av.open(str(out / "replay.mp4"), mode="w")
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height = 640, 480
        stream.pix_fmt = "yuv420p"
        done = False
        with torch.no_grad():
            while not done:
                frame = av.VideoFrame.from_ndarray(env.render(), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                action = policy(torch.as_tensor(obs, dtype=torch.float32)).numpy()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        env.close()


if __name__ == "__main__":
    main()
