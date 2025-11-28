import cv2
import numpy as np
from typing import Tuple, Optional
import os
import matplotlib.pyplot as plt

def read_txt_points(txt_path: str, dim: int) -> Optional[np.ndarray]:
    """
    从txt文件读取2D或3D点数据
    Args:
        txt_path: txt文件路径（每行对应一个点，数值用空格/制表符分隔）
        dim: 点的维度（2=2D点，3=3D点）
    Returns: 形状为(N, dim)的ndarray，N为点的数量；读取失败返回None
    """
    try:
        # 读取txt文件，自动处理分隔符
        data = np.loadtxt(txt_path, delimiter=None, dtype=np.float32)

        # 处理一维数据（仅1个点时）
        if data.ndim == 1:
            data = data.reshape(1, dim)

        # 校验数据维度
        assert data.shape[1] == dim, f"数据维度错误：需{dim}列，实际{data.shape[1]}列"

        # 去重（避免重复点影响求解）
        data = np.unique(data, axis=0)

        print(f"成功读取 {txt_path}：{data.shape[0]} 个{dim}D点")
        return data
    except FileNotFoundError:
        print(f"错误：文件 {txt_path} 不存在")
        return None
    except AssertionError as e:
        print(f"错误：{str(e)}")
        return None
    except Exception as e:
        print(f"读取文件失败：{str(e)}")
        return None


def solve_pnp_opencv(
        img_2d_points_path: str,
        pcd_3d_points_path: str,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        image_path: str = "image.png",
        output_vis_path: str = "reprojection_on_image.png"
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:

    # ========== 1. 读取数据 ==========
    img_points = read_txt_points(img_2d_points_path, dim=2)
    obj_points = read_txt_points(pcd_3d_points_path, dim=3)

    if img_points is None or obj_points is None:
        return None, None, None
    if img_points.shape[0] != obj_points.shape[0]:
        print("2D 和 3D 点数量不一致！")
        return None, None, None

    # ========== 2. 相机参数 ==========
    if camera_matrix is None:
        fx, fy = 1000.0, 1000.0
        cx, cy = 960.0, 540.0
        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float32)
        print("⚠ 使用默认相机内参")

    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        print("⚠ 使用无畸变模型")

    # ========== 3. PnP ==========
    ret, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_DLS
    )
    if not ret:
        print("PnP 求解失败")
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = tvec.squeeze()

    # ========== 4. 重投影 ==========
    reproj_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj_points = reproj_points.squeeze()

    # 逐点误差
    per_point_errors = np.linalg.norm(img_points - reproj_points, axis=1)
    mean_error = np.mean(per_point_errors)

    print("平均误差 = ", mean_error)

    # ========== 5. 读取图像并绘制结果 ==========
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像：{image_path}")
        return rvec, tvec, transform_matrix

    # OpenCV 坐标为 (x, y)
    for i in range(len(img_points)):
        x1, y1 = img_points[i]
        x2, y2 = reproj_points[i]

        # 原始点（蓝色）
        cv2.circle(img, (int(x1), int(y1)), 3, (255, 0, 0), -1)

        # 重投影点（红色）
        cv2.circle(img, (int(x2), int(y2)), 3, (0, 0, 255), -1)

        # 误差连线（黄色）
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        # 标出误差数值
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        cv2.putText(img, f"{per_point_errors[i]:.2f}",
                    (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    # ========== 6. 绘制误差直方图并保存 ==========
    # 使用 matplotlib 绘制第二张图（直方图）
    plt.figure(figsize=(8, 4))
    plt.hist(per_point_errors, bins=10, alpha=0.7)
    plt.title(f"Reprojection Error Histogram\nMean = {mean_error:.3f}px")
    plt.xlabel("Error (px)")
    plt.ylabel("Count")
    hist_path = output_vis_path.replace(".png", "_hist.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()

    cv2.namedWindow("重投影误差",cv2.WINDOW_NORMAL)
    cv2.imshow("重投影误差",img)
    cv2.waitKey()
    # ========== 7. 保存叠加图像 ==========
    cv2.imwrite(output_vis_path, img)
    print(f"重投影可视化已保存：{output_vis_path}")
    print(f"误差直方图已保存：{hist_path}")

    return rvec, tvec, transform_matrix

def save_transform_matrix(transform_matrix: np.ndarray, output_path: str = "transform_matrix.txt") -> None:
    """
    保存4×4变换矩阵到txt文件
    Args:
        transform_matrix: 4×4变换矩阵
        output_path: 输出txt文件路径
    """
    np.savetxt(
        output_path,
        transform_matrix,
        fmt="%.6f",  # 保留6位小数
        delimiter=" ",
        header="4x4 Transform Matrix (World → Camera): [R | t; 0 0 0 1]",
        comments=""
    )
    print(f"\n变换矩阵已保存至：{output_path}")


# ---------------------- 示例调用 ----------------------
if __name__ == "__main__":
    # 1. 配置输入输出路径（替换为你的文件路径）
    IMG_2D_POINTS_PATH = "pnp_solve/circle_centers_images.txt"  # 2D图像点txt（N×2，格式：x y）
    PCD_3D_POINTS_PATH = "pnp_solve/circle_centers_points.txt"  # 3D点云点txt（N×3，格式：x y z）
    OUTPUT_MATRIX_PATH = "transform_matrix.txt"  # 变换矩阵输出路径
    OUTPUT_PNG_PATH = "./results/reprojection_error.png"  # 输出PNG路径（必填）
    INPUT_IMAGE = "./waican_data/20250314_220414(14).jpg"
    # 2. （可选）传入真实相机内参（替换默认值）
    # 假设已通过相机标定得到内参，示例：
    real_camera_matrix = np.array([[577.6673, 0, 188.6225],
                                   [0, 577.4241, 148.4147],
                                   [0, 0, 1]], dtype=np.float32)
    real_dist_coeffs = np.array([[-0.3404, 0.3301, 0, 0, 0]], dtype=np.float32)

    # 3. 求解PnP
    rvec, tvec, transform_matrix = solve_pnp_opencv(
        img_2d_points_path=IMG_2D_POINTS_PATH,
        pcd_3d_points_path=PCD_3D_POINTS_PATH,
        output_vis_path=OUTPUT_PNG_PATH,  # 传入PNG保存路径
        camera_matrix=real_camera_matrix,  # 启用真实内参
        dist_coeffs=real_dist_coeffs,       # 启用真实畸变系数
        image_path=INPUT_IMAGE
    )

    # 4. 保存变换矩阵
    if transform_matrix is not None:
        save_transform_matrix(transform_matrix, OUTPUT_MATRIX_PATH)