# 点云脚本(包含了点云处理的各种脚本)

## 1.点云脚本描述
- transform.py:输入：点云数据（ply格式）和变换矩阵（txt格式），输出：点云数据（ply格式）
- voxel_subsample.py：对输入点云进行下采样和尺度变换（先进行下采样再进行尺度变换）
- voxel_subsample_1.py：对输入点云进行下采样和尺度变换（先进行尺度变换再进行下采样）
- pre_analyse.py：对点云的变换矩阵进行精度分析