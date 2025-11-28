import open3d as o3d
import numpy as np
import cv2
from PIL import Image
import argparse


def read_extrinsic_from_txt(txt_path):
    """
    从TXT文件读取4×4外参矩阵
    Args:
        txt_path (str): 外参TXT文件路径
    Returns:
        np.ndarray: 4×4外参变换矩阵
    """
    # 读取TXT文件，按空格/换行分割数据，过滤空字符，转换为float类型
    with open(txt_path, 'r', encoding='utf-8') as f:
        data = f.read().split()  # 分割所有数字（支持空格、换行分隔）
        data = [float(x) for x in data if x.strip()]  # 去除空字符并转浮点数

    # 验证数据长度是否为16（4×4矩阵共16个元素）
    if len(data) != 16:
        raise ValueError(f"TXT文件中数据个数错误！需16个元素（4×4矩阵），实际读取到{len(data)}个")

    # 转换为4×4 numpy矩阵（按行优先排列，符合常规矩阵存储习惯）
    extrinsic_matrix = np.array(data).reshape(4, 4)
    print(f"成功从{txt_path}读取外参矩阵：")
    print(extrinsic_matrix)
    return extrinsic_matrix


def project_pointcloud_to_image(ply_path, img_path, extrinsic_matrix, intrinsic_matrix,
                                draw_color='point_color', point_size=2):
    """
    点云映射到图像的核心函数
    Args:
        ply_path (str): PLY点云文件路径
        img_path (str): 原始JPG图像路径
        extrinsic_matrix (np.ndarray): 外参变换矩阵 (4x4, 世界坐标系→相机坐标系)
        intrinsic_matrix (np.ndarray): 相机内参矩阵 (3x3)
        draw_color (str): 绘制颜色模式 ('point_color'=点云自身颜色, 'depth'=深度编码颜色)
        point_size (int): 投影点绘制大小
    Returns:
        PIL.Image: 映射后的图像对象
    """
    # -------------------------- 1. 读取点云和图像 --------------------------
    # 读取PLY点云（Open3D支持彩色/非彩色点云）
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        raise ValueError("点云文件中没有有效点数据")

    # 提取3D点坐标 (N×3, N为点云数量)
    points_3d = np.asarray(pcd.points)
    # 提取点云颜色（若有，N×3，范围[0,1]）
    # has_color = pcd.has_colors()
    # if has_color:
    #     points_color = np.asarray(pcd.colors) * 255  # 转换为0-255范围（适配OpenCV）
    # else:
    #     print("警告：点云无颜色信息，将使用深度编码颜色")
    #     draw_color = 'depth'  # 强制切换为深度模式

    # 读取原始图像（OpenCV读取为BGR格式，后续需转换为RGB）
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像文件：{img_path}")
    img_height, img_width = img.shape[:2]
    print(f"图像尺寸：{img_width}×{img_height}")

    # -------------------------- 2. 点云坐标变换与投影 --------------------------
    # 步骤1：将3D点从世界坐标系转换到相机坐标系（外参应用）
    # 给点云添加齐次坐标 (N×3 → N×4, 最后一维补1)
    points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    # 外参矩阵：世界→相机（格式：[R | t]; [0 0 0 1]，R为3×3旋转矩阵，t为3×1平移向量）
    points_cam_homo = points_3d_homo @ extrinsic_matrix.T  # N×4 → 矩阵乘法（注意转置）
    points_cam = points_cam_homo[:, :3]  # 去除齐次坐标，得到相机坐标系下的3D点 (N×3)

    # 步骤2：过滤无效点（相机坐标系下z<0：点在相机后方，无法投影）
    valid_mask = points_cam[:, 2] > 1e-3  # z>0且不为极小值（避免除以0）
    points_cam_valid = points_cam[valid_mask]
    # if has_color:
    #     points_color_valid = points_color[valid_mask].astype(np.int32)
    print(f"有效投影点数量：{len(points_cam_valid)}/{len(points_3d)}")

    # 步骤3：相机坐标系→图像平面（内参应用，透视投影）
    # 透视投影公式：u = (fx * X + cx * Z) / Z, v = (fy * Y + cy * Z) / Z
    # 简化为：[u*Z; v*Z; Z] = intrinsic_matrix @ [X; Y; Z]
    proj_homo = intrinsic_matrix @ points_cam_valid.T  # 3×N（每列对应一个点的投影齐次坐标）
    Z = proj_homo[2, :]  # 每个点的深度（相机坐标系下z值）
    u = proj_homo[0, :] / Z  # 图像平面x坐标（列）
    v = proj_homo[1, :] / Z  # 图像平面y坐标（行）

    # 步骤4：过滤超出图像范围的点（u∈[0, img_width), v∈[0, img_height)）
    img_mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    u_valid = u[img_mask].astype(np.int32)
    v_valid = v[img_mask].astype(np.int32)
    Z_valid = Z[img_mask]
    # if has_color:
    #     points_color_valid = points_color_valid[img_mask]
    print(f"图像范围内的投影点数量：{len(u_valid)}")

    # -------------------------- 3. 绘制投影点到图像 --------------------------
    # 复制原始图像，避免修改原图
    img_projected = img.copy()
    white = (255, 255, 255)  # 白色RGB值（OpenCV为BGR格式，此处顺序对应）

    # 计算像素块的半边长（使投影点位于块中心）
    half_size = point_size // 2

    for i in range(len(u_valid)):
        # 当前投影点的中心坐标（x=列，y=行）
        x_center = u_valid[i]
        y_center = v_valid[i]

        # 计算白色像素块的边界（避免超出图像范围）
        x_start = max(0, x_center - half_size)
        x_end = min(img_width - 1, x_center + half_size)
        y_start = max(0, y_center - half_size)
        y_end = min(img_height - 1, y_center + half_size)

        # 直接修改图像矩阵：将矩形区域的所有像素设为白色
        # 注意OpenCV图像格式为 (行, 列, 通道) → [y_start:y_end, x_start:x_end]
        img_projected[y_start:y_end + 1, x_start:x_end + 1, :] = white

    # -------------------------- 4. 格式转换与返回 --------------------------
    img_projected_rgb = cv2.cvtColor(img_projected, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_projected_rgb)


if __name__ == "__main__":
    # 解析命令行参数（新增--extrinsic_txt参数，指定外参TXT路径）
    parser = argparse.ArgumentParser(description="点云PLY映射到JPG图像（支持从TXT读取外参）")
    parser.add_argument("--ply_path", default="./point_cloud_deal/output_target_cluster.ply",help="PLY点云文件路径")
    parser.add_argument("--img_path", default="./waican_data/20250314_220414(14).jpg",help="原始JPG图像路径")
    parser.add_argument("--extrinsic_txt", default="./transform_matrix.txt", help="外参矩阵TXT文件路径（4×4格式）")
    parser.add_argument("--output_path", default="projected_image.jpg", help="输出映射图像路径")
    args = parser.parse_args()

    # -------------------------- 配置参数（需根据实际场景修改） --------------------------
    # 1. 相机内参矩阵 K (3x3)：[fx, 0, cx; 0, fy, cy; 0, 0, 1]
    # fx/fy：x/y轴焦距（像素单位）；cx/cy：主点坐标（通常为图像中心）
    # 来源：相机标定结果（如OpenCV标定工具、ROS标定包）
    intrinsic_matrix = np.array([
        [577.6673, 0.0, 188.6225],  # fx=1000, cx=500（图像宽度1000时的中心）
        [0.0, 577.4241, 148.4147],  # fy=1000, cy=375（图像高度750时的中心）
        [0.0, 0.0, 1.0]
    ])

    # 2. 从TXT文件读取外参矩阵（核心修改点）
    try:
        extrinsic_matrix = read_extrinsic_from_txt(args.extrinsic_txt)
    except Exception as e:
        print(f"读取外参失败：{str(e)}")
        exit(1)

    # -------------------------- 执行映射并保存 --------------------------
    try:
        # 调用映射函数
        projected_img = project_pointcloud_to_image(
            ply_path=args.ply_path,
            img_path=args.img_path,
            extrinsic_matrix=extrinsic_matrix,
            intrinsic_matrix=intrinsic_matrix,
            draw_color='point_color',  # 可选：'point_color'（点云颜色）或 'depth'（深度编码）
            point_size=1  # 投影点大小（像素），可根据图像分辨率调整
        )
        # 保存输出图像
        projected_img.save(args.output_path)
        print(f"\n映射图像已成功保存到：{args.output_path}")
        # 可选：显示映射后的图像
        projected_img.show()
    except Exception as e:
        print(f"执行映射失败：{str(e)}")