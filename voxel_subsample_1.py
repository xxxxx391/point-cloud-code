import open3d as o3d
import numpy as np

def downsample_ply(pcd, voxel_size=1, output_file_path = "1.ply"):
    """
    从 PLY 文件中读取点云数据并进行体素下采样
    :param file_path: PLY 文件的路径
    :param voxel_size: 体素的大小，用于下采样，默认为 0.05
    :return: 下采样后的点云对象
    """
    try:
        # 从 PLY 文件中读取点云数据
        # pcd = o3d.io.read_point_cloud(file_path)
        # print(f"成功读取点云数据，原始点数: {len(pcd.points)}")

        # 进行体素下采样
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"下采样后点数: {len(downsampled_pcd.points)}")

        o3d.io.write_point_cloud(output_file_path, downsampled_pcd)

        return downsampled_pcd
    except Exception as e:
        print(f"读取文件时出现错误: {e}")
        return None

def save_pcd_to_ply(pcd, output_file_path):
    """
    将点云对象保存为 PLY 格式文件
    :param pcd: 点云对象
    :param output_file_path: 输出 PLY 文件的路径
    """
    try:
        # 保存点云对象为 PLY 文件
        o3d.io.write_point_cloud(output_file_path, pcd)
        print(f"成功将点云保存到 {output_file_path}")
    except Exception as e:
        print(f"保存文件时出现错误: {e}")

def scale_point_cloud(pcd):
    """
    读取点云文件，将点云坐标缩小100倍，并保留三位小数，然后保存为新的点云文件
    :param input_file_path: 输入点云文件的路径
    :param output_file_path: 输出点云文件的路径
    """
    try:
        # # 从指定文件路径读取点云数据
        # pcd = o3d.io.read_point_cloud(input_file_path)

        # 获取点云的坐标数组
        points = np.asarray(pcd.points)

        # 将坐标缩小100倍
        scaled_points = points / 100

        # 保留三位小数
        scaled_points = np.round(scaled_points, 3)

        # 更新点云对象的坐标
        pcd.points = o3d.utility.Vector3dVector(scaled_points)

        # 将处理后的点云保存为新的PLY文件
        # o3d.io.write_point_cloud(output_file_path, pcd)

        print(f"点云坐标已成功缩小100倍并保留三位小数")

        return pcd
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    # 替换为你的 PLY 文件路径
    ply_file_path = "xixihe_UAV - Cloud - Cloud.ply"
    output_file_path = "UAV.ply"
    pcd = o3d.io.read_point_cloud(ply_file_path)

    ply_file_path_s = scale_point_cloud(pcd)
    # 调用函数进行读取和下采样
    downsampled_cloud = downsample_ply(ply_file_path_s,voxel_size = 0.02, output_file_path = output_file_path)

    if downsampled_cloud:
        # scale_point_cloud(downsampled_cloud,"TLS.ply")
        # 可视化下采样后的点云
        o3d.visualization.draw_geometries([downsampled_cloud])