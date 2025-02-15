import numpy as np
import open3d as o3d

def read_transform_matrix_from_txt(file_path):
    """
    从txt文件中读取变换矩阵
    :param file_path: txt文件路径
    :return: 4x4的变换矩阵
    """
    transform_matrix = np.loadtxt(file_path)
    if transform_matrix.shape != (4, 4):
        raise ValueError("变换矩阵的形状必须是4x4")
    return transform_matrix

def read_point_cloud_from_pcd(file_path):
    """
    从pcd文件中读取点云数据
    :param file_path: pcd文件路径
    :return: 点云对象
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def apply_transform(pcd, transform_matrix,output_file_path):
    """
    将变换矩阵应用到点云数据上
    :param pcd: 点云对象
    :param transform_matrix: 4x4的变换矩阵
    :return: 变换后的点云对象
    """
    transformed_pcd = pcd.transform(transform_matrix)
    o3d.io.write_point_cloud(output_file_path, pcd)
    return transformed_pcd

def visualize_point_clouds(original_pcd, transformed_pcd):
    """
    可视化变换前后的点云数据
    :param original_pcd: 原始点云对象
    :param transformed_pcd: 变换后的点云对象
    """
    original_pcd.paint_uniform_color([1, 0, 0])  # 原始点云设为红色
    transformed_pcd.paint_uniform_color([0, 1, 0])  # 变换后的点云设为绿色
    o3d.visualization.draw_geometries([original_pcd, transformed_pcd])

if __name__ == "__main__":
    # 替换为你的pcd文件路径
    pcd_file_path = "UAV.ply"
    out_file = "dealt_data/UAV_RT.ply"
    # 替换为你的txt文件路径
    txt_file_path = "transformation_matrix.txt"

    # 读取点云数据
    original_pcd = read_point_cloud_from_pcd(pcd_file_path)
    # 读取变换矩阵
    transform_matrix = read_transform_matrix_from_txt(txt_file_path)
    # 应用变换矩阵
    transformed_pcd = apply_transform(original_pcd, transform_matrix,out_file)
    # 可视化变换前后的点云
    visualize_point_clouds(original_pcd, transformed_pcd)