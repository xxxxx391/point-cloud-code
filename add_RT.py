import open3d as o3d
import numpy as np

def read_point_cloud(file_path):
    """
    从指定文件路径读取点云数据
    :param file_path: 点云文件的路径
    :return: 读取到的点云对象
    """
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        print(f"成功读取点云数据，点数: {len(pcd.points)}")
        return pcd
    except Exception as e:
        print(f"读取文件时出现错误: {e}")
        return None

def get_rotation_matrix(rotation_angle, rotation_axis):
    """
    根据旋转角度和旋转轴计算旋转矩阵
    :param rotation_angle: 旋转角度（弧度）
    :param rotation_axis: 旋转轴向量
    :return: 旋转矩阵
    """
    # 归一化旋转轴
    rotation_axis = np.array(rotation_axis) / np.linalg.norm(rotation_axis)
    n = rotation_axis
    # 计算反对称矩阵
    n_cross = np.array([
        [0, -n[2], n[1]],
        [n[2], 0, -n[0]],
        [-n[1], n[0], 0]
    ])
    # 计算旋转矩阵
    R = np.cos(rotation_angle) * np.eye(3) + (1 - np.cos(rotation_angle)) * np.outer(n, n) + np.sin(rotation_angle) * n_cross
    return R

def rotate_point_cloud(pcd, rotation_angle, rotation_axis):
    """
    对输入的点云数据进行旋转操作
    :param pcd: 输入的点云对象
    :param rotation_angle: 旋转角度（弧度）
    :param rotation_axis: 旋转轴，例如 [1, 0, 0] 表示绕 x 轴旋转
    :return: 旋转后的点云对象和旋转矩阵
    """
    R = get_rotation_matrix(rotation_angle, rotation_axis)
    pcd.rotate(R, center=pcd.get_center())
    return pcd, R

def translate_point_cloud(pcd, translation):
    """
    对输入的点云数据进行平移操作
    :param pcd: 输入的点云对象
    :param translation: 平移向量，例如 [1, 2, 3] 表示在 x、y、z 方向分别平移 1、2、3 个单位
    :return: 平移后的点云对象和平移矩阵
    """
    T = np.eye(4)
    T[:3, 3] = np.array(translation)
    pcd.translate(np.array(translation))
    return pcd, T

def save_point_cloud(pcd, output_file_path):
    """
    将处理后的点云数据保存为 PLY 文件
    :param pcd: 处理后的点云对象
    :param output_file_path: 输出文件的路径
    """
    try:
        o3d.io.write_point_cloud(output_file_path, pcd)
        print(f"成功将点云保存到 {output_file_path}")
    except Exception as e:
        print(f"保存文件时出现错误: {e}")

def save_transformation_matrix(matrix, matrix_file_path):
    """
    保存变换矩阵到 .txt 文件
    :param matrix: 变换矩阵
    :param matrix_file_path: 矩阵文件保存路径
    """
    try:
        np.savetxt(matrix_file_path, matrix, fmt='%.8f', delimiter=' ')
        print(f"成功将变换矩阵保存到 {matrix_file_path}")
    except Exception as e:
        print(f"保存变换矩阵时出现错误: {e}")

def visualize_point_cloud(pcd):
    """
    可视化点云数据
    :param pcd: 要可视化的点云对象
    """
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # 替换为你的 PLY 文件路径
    input_file_path = "TLS.ply"
    # 替换为你希望保存的 PLY 文件路径
    output_file_path = "TLS_RT.ply"
    # 替换为你希望保存变换矩阵的 .txt 文件路径
    matrix_file_path = "transformation_matrix.txt"

    # 读取点云数据
    pcd = read_point_cloud(input_file_path)
    if pcd is not None:
        # 定义旋转角度（这里以绕 z 轴旋转 45 度为例，需将角度转换为弧度）
        rotation_angle = np.radians(45)
        # 定义旋转轴，[0, 0, 1] 表示绕 z 轴旋转
        rotation_axis = [0, 0, 1]
        # 进行旋转操作
        pcd, rotation_matrix = rotate_point_cloud(pcd, rotation_angle, rotation_axis)

        # 定义平移向量，例如在 x 方向平移 2 个单位，y 方向平移 3 个单位，z 方向平移 4 个单位
        translation = [1.2, 0.5, 1.6]
        # 进行平移操作
        pcd, translation_matrix = translate_point_cloud(pcd, translation)

        # 计算总的变换矩阵
        rotation_matrix_homogeneous = np.vstack((np.hstack((rotation_matrix, np.zeros((3, 1)))), [0, 0, 0, 1]))
        combined_matrix = np.dot(translation_matrix, rotation_matrix_homogeneous)

        # 保存处理后的点云数据
        save_point_cloud(pcd, output_file_path)

        # 保存变换矩阵
        save_transformation_matrix(combined_matrix, matrix_file_path)

        # 可视化处理后的点云数据
        visualize_point_cloud(pcd)