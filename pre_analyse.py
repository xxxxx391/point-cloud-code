import numpy as np

def read_transform_matrix(file_path):
    """
    从 txt 文件中读取 4x4 的变换矩阵
    :param file_path: txt 文件的路径
    :return: 4x4 的 numpy 数组，表示变换矩阵
    """
    matrix = np.loadtxt(file_path)
    if matrix.shape != (4, 4):
        raise ValueError("输入的矩阵不是 4x4 的变换矩阵")
    return matrix

def calculate_translation_error(gt_matrix, calculated_matrix):
    """
    计算平移误差，即两个变换矩阵平移部分的欧几里得距离
    :param gt_matrix: 真实的变换矩阵
    :param calculated_matrix: 计算得到的变换矩阵
    :return: 平移误差
    """
    gt_translation = gt_matrix[:3, 3]
    calculated_translation = calculated_matrix[:3, 3]
    translation_error = np.linalg.norm(gt_translation - calculated_translation)
    return translation_error

def calculate_rotation_error(gt_matrix, calculated_matrix):
    """
    计算旋转误差，通过计算两个旋转矩阵之间的角度误差
    :param gt_matrix: 真实的变换矩阵
    :param calculated_matrix: 计算得到的变换矩阵
    :return: 旋转误差（弧度）
    """
    gt_rotation = gt_matrix[:3, :3]
    calculated_rotation = calculated_matrix[:3, :3]
    rotation_difference = np.dot(gt_rotation.T, calculated_rotation)
    trace = np.trace(rotation_difference)
    cos_angle = (trace - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)  # 确保在 [-1, 1] 范围内
    rotation_error = np.arccos(cos_angle)
    return rotation_error

def analyze_registration_accuracy(gt_file_path, calculated_file_path):
    """
    分析点云配准的精度
    :param gt_file_path: 存储真实变换矩阵的 txt 文件路径
    :param calculated_file_path: 存储计算得到的变换矩阵的 txt 文件路径
    :return: 平移误差和旋转误差
    """
    # 读取真实变换矩阵和计算得到的变换矩阵
    gt_matrix = read_transform_matrix(gt_file_path)
    calculated_matrix = read_transform_matrix(calculated_file_path)

    # 计算平移误差和旋转误差
    translation_error = calculate_translation_error(gt_matrix, calculated_matrix)
    rotation_error = calculate_rotation_error(gt_matrix, calculated_matrix)

    return translation_error, rotation_error

if __name__ == "__main__":
    # 替换为实际的文件路径
    gt_file_path = "compute_transform.txt"
    calculated_file_path = "transformation_matrix.txt"

    # 分析配准精度
    translation_error, rotation_error = analyze_registration_accuracy(gt_file_path, calculated_file_path)

    # 输出分析结果
    print(f"平移误差: {translation_error}")
    print(f"旋转误差: {rotation_error} 弧度")
    print(f"旋转误差: {np.rad2deg(rotation_error)} 度")