import argparse
import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle Adjustment with PyTorch")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--iters", type=int, default=3000, help="Number of optimization iterations")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--fov-deg", type=float, default=60.0, help="Initial field-of-view in degrees")
    parser.add_argument("--init-depth", type=float, default=2.5, help="Initial camera depth d, Tz=-d")
    parser.add_argument("--point-scale", type=float, default=0.25, help="Std for 3D point init")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--print-every", type=int, default=100, help="Logging frequency")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles to rotation matrices.

    Args:
        euler: (V, 3), angles in radians.
    Returns:
        R: (V, 3, 3)
    """
    ex, ey, ez = euler[:, 0], euler[:, 1], euler[:, 2]

    cx, sx = torch.cos(ex), torch.sin(ex)
    cy, sy = torch.cos(ey), torch.sin(ey)
    cz, sz = torch.cos(ez), torch.sin(ez)

    ones = torch.ones_like(ex)
    zeros = torch.zeros_like(ex)

    rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cx, -sx], dim=-1),
            torch.stack([zeros, sx, cx], dim=-1),
        ],
        dim=-2,
    )
    ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    # XYZ convention: R = Rz @ Ry @ Rx when applying column vectors with right-multiplication equivalent.
    return rz @ ry @ rx


def load_observations(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points2d_npz = np.load(os.path.join(data_dir, "points2d.npz"))
    keys = sorted(points2d_npz.files)
    obs = np.stack([points2d_npz[k] for k in keys], axis=0)  # (V, N, 3)
    points2d = obs[:, :, :2].astype(np.float32)
    visibility = obs[:, :, 2].astype(np.float32)
    colors = np.load(os.path.join(data_dir, "points3d_colors.npy")).astype(np.float32)
    return points2d, visibility, colors


def project_points(
    points3d: torch.Tensor,
    euler: torch.Tensor,
    trans: torch.Tensor,
    focal: torch.Tensor,
    cx: float,
    cy: float,
) -> torch.Tensor:
    """Project world points to pixels for all views.

    points3d: (N, 3)
    euler: (V, 3)
    trans: (V, 3)
    focal: scalar tensor
    returns: (V, N, 2)
    """
    rmat = euler_xyz_to_matrix(euler)  # (V, 3, 3)
    # Xc = R @ X + T
    xc = torch.einsum("vij,nj->vni", rmat, points3d) + trans[:, None, :]  # (V, N, 3)

    z = xc[:, :, 2]
    eps = 1e-6
    # Keep the sign of depth and only avoid division by zero.
    z = torch.where(z >= 0, torch.clamp(z, min=eps), torch.clamp(z, max=-eps))
    u = -focal * (xc[:, :, 0] / z) + cx
    v = focal * (xc[:, :, 1] / z) + cy
    return torch.stack([u, v], dim=-1)


def save_colored_obj(path: str, points3d: np.ndarray, colors: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p, c in zip(points3d, colors):
            r, g, b = float(c[0]), float(c[1]), float(c[2])
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r:.6f} {g:.6f} {b:.6f}\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    points2d_np, visibility_np, colors_np = load_observations(args.data_dir)
    num_views, num_points, _ = points2d_np.shape

    image_width = 1024.0
    image_height = 1024.0
    cx = image_width / 2.0
    cy = image_height / 2.0

    points2d = torch.from_numpy(points2d_np).to(device)
    visibility = torch.from_numpy(visibility_np).to(device)

    # FOV-based focal initialization.
    f_init = image_height / (2.0 * math.tan(math.radians(args.fov_deg) / 2.0))
    log_f = torch.nn.Parameter(torch.tensor(math.log(f_init), dtype=torch.float32, device=device))

    euler = torch.nn.Parameter(torch.zeros((num_views, 3), dtype=torch.float32, device=device))
    trans = torch.nn.Parameter(torch.zeros((num_views, 3), dtype=torch.float32, device=device))
    with torch.no_grad():
        trans[:, 2] = -float(args.init_depth)

    points3d = torch.nn.Parameter(
        torch.randn((num_points, 3), dtype=torch.float32, device=device) * float(args.point_scale)
    )

    optimizer = torch.optim.Adam([log_f, euler, trans, points3d], lr=args.lr)
    losses = []

    for it in range(1, args.iters + 1):
        optimizer.zero_grad(set_to_none=True)

        focal = torch.exp(log_f)
        pred = project_points(points3d, euler, trans, focal, cx, cy)  # (V, N, 2)

        diff = pred - points2d
        sq = (diff * diff).sum(dim=-1)  # (V, N)
        weighted = sq * visibility
        loss = weighted.sum() / visibility.sum().clamp(min=1.0)

        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if it % args.print_every == 0 or it == 1 or it == args.iters:
            print(f"[Iter {it:04d}/{args.iters}] loss={loss.item():.6f}, f={torch.exp(log_f).item():.3f}")

    # Save curve.
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, len(losses) + 1), losses, linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Reprojection MSE")
    plt.title("Bundle Adjustment Optimization Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_png = os.path.join(args.out_dir, "ba_loss.png")
    plt.savefig(loss_png, dpi=180)
    plt.close()

    # Save optimized values.
    focal_val = float(torch.exp(log_f).detach().cpu().item())
    np.savez(
        os.path.join(args.out_dir, "ba_params.npz"),
        focal=np.array([focal_val], dtype=np.float32),
        euler=euler.detach().cpu().numpy().astype(np.float32),
        trans=trans.detach().cpu().numpy().astype(np.float32),
        loss=np.array(losses, dtype=np.float32),
    )

    # Save colored point cloud as OBJ.
    obj_path = os.path.join(args.out_dir, "reconstruction.obj")
    save_colored_obj(obj_path, points3d.detach().cpu().numpy().astype(np.float32), colors_np)

    print("Done.")
    print(f"- loss curve: {loss_png}")
    print(f"- point cloud: {obj_path}")
    print(f"- params: {os.path.join(args.out_dir, 'ba_params.npz')}")


if __name__ == "__main__":
    main()
