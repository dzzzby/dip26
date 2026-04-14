import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    if image is None:
        return None

    warped_image = np.array(image)
    h, w = warped_image.shape[:2]

    source_pts = np.asarray(source_pts, dtype=np.float64)
    target_pts = np.asarray(target_pts, dtype=np.float64)

    n = min(len(source_pts), len(target_pts))
    if n == 0:
        return warped_image

    source_pts = source_pts[:n]
    target_pts = target_pts[:n]

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    query = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # Backward mapping: estimate where each target pixel comes from in source image.
    if n < 3:
        disp = source_pts - target_pts
        diff = query[:, None, :] - target_pts[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)

        if n == 1:
            weighted_disp = np.repeat(disp, query.shape[0], axis=0)
        else:
            weights = 1.0 / np.power(dist2 + eps, max(alpha, 1e-6) / 2.0)
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            weights = weights / np.maximum(weights_sum, eps)
            weighted_disp = weights @ disp

        mapped = query + weighted_disp
        map_x = mapped[:, 0].reshape(h, w).astype(np.float32)
        map_y = mapped[:, 1].reshape(h, w).astype(np.float32)

        return cv2.remap(
            warped_image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    # Thin Plate Spline radial basis: U(r) = r^2 log(r^2 + eps)
    diff_ctrl = target_pts[:, None, :] - target_pts[None, :, :]
    r2_ctrl = np.sum(diff_ctrl * diff_ctrl, axis=2)
    K = r2_ctrl * np.log(r2_ctrl + eps)

    # alpha controls smoothness regularization.
    reg = max(alpha, 0.0) * 1e-3
    K = K + reg * np.eye(n, dtype=np.float64)

    P = np.concatenate([np.ones((n, 1), dtype=np.float64), target_pts], axis=1)

    L = np.zeros((n + 3, n + 3), dtype=np.float64)
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T

    Yx = np.concatenate([source_pts[:, 0], np.zeros(3, dtype=np.float64)])
    Yy = np.concatenate([source_pts[:, 1], np.zeros(3, dtype=np.float64)])

    params_x = np.linalg.solve(L, Yx)
    params_y = np.linalg.solve(L, Yy)

    w_x, a_x = params_x[:n], params_x[n:]
    w_y, a_y = params_y[:n], params_y[n:]

    diff_query = query[:, None, :] - target_pts[None, :, :]
    r2_query = np.sum(diff_query * diff_query, axis=2)
    U_query = r2_query * np.log(r2_query + eps)
    P_query = np.concatenate([np.ones((query.shape[0], 1), dtype=np.float64), query], axis=1)

    map_x = (U_query @ w_x + P_query @ a_x).reshape(h, w).astype(np.float32)
    map_y = (U_query @ w_y + P_query @ a_y).reshape(h, w).astype(np.float32)

    warped_image = cv2.remap(
        warped_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return warped_image

def run_warping():
    global points_src, points_dst, image

    if image is None:
        return None

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
