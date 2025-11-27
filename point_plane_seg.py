# -*-coding:utf-8 -*-
import os
import open3d as o3d
import numpy as np
# from scipy.optimize import least_squares  # 保留原导入（若后续扩展使用）
import matplotlib.pyplot as plt
from collections import Counter
from typing import Tuple, Optional, Union


class PointCloudProcessor:
    """
    点云处理工具类：实现点云读取、平面分割、DBSCAN聚类、高频聚类筛选及结果保存
    """
    def __init__(self, input_pcd_path: str, output_dir: str = "./点云特征提取数据"):
        """
        初始化处理器
        Args:
            input_pcd_path: 输入点云文件路径（.pcd格式）
            output_dir: 输出文件保存目录（默认：./点云特征提取数据）
        """
        self.input_pcd_path = input_pcd_path
        self.output_dir = output_dir
        # 初始化核心数据（后续处理结果将存储于此）
        self.raw_pcd: Optional[o3d.geometry.PointCloud] = None
        self.plane_model: Optional[np.ndarray] = None
        self.inlier_cloud: Optional[o3d.geometry.PointCloud] = None
        self.outlier_cloud: Optional[o3d.geometry.PointCloud] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.target_cluster_cloud: Optional[o3d.geometry.PointCloud] = None

        # 创建输出目录（若不存在）
        os.makedirs(self.output_dir, exist_ok=True)

    def get_most_frequent_number_indices(self, arr: Union[list, np.ndarray]) -> np.ndarray:
        """
        辅助函数：获取数组中出现次数最多元素的所有索引
        Args:
            arr: 输入数组（list或ndarray）
        Returns: 高频元素的索引数组
        """
        np_arr = np.array(arr)
        counter = Counter(np_arr)
        most_common_element = counter.most_common(1)[0][0]
        return np.where(np_arr == most_common_element)[0]

    def load_point_cloud(self) -> bool:
        """
        读取点云文件
        Returns: 读取成功返回True，失败返回False
        """
        try:
            self.raw_pcd = o3d.io.read_point_cloud(self.input_pcd_path)
            print(f"成功读取点云文件：{self.input_pcd_path}")
            print(f"点云总点数：{len(self.raw_pcd.points)}")
            return True
        except Exception as e:
            print(f"点云读取失败：{str(e)}")
            return False

    def segment_plane(self, distance_threshold: float = 3.0, ransac_n: int = 3, num_iterations: int = 1000) -> bool:
        """
        平面分割：提取点云中的主平面（内点）和非平面（外点）
        Args:
            distance_threshold: 点到平面的距离阈值（默认：3.0）
            ransac_n: RANSAC算法每次采样的点数（默认：3）
            num_iterations: RANSAC迭代次数（默认：1000）
        Returns: 分割成功返回True，失败返回False
        """
        if self.raw_pcd is None:
            print("请先调用load_point_cloud()读取点云")
            return False

        try:
            self.plane_model, inliers = self.raw_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            # 提取平面内点和外点
            self.inlier_cloud = self.raw_pcd.select_by_index(inliers)
            self.outlier_cloud = self.raw_pcd.select_by_index(inliers, invert=True)
            # 打印平面方程
            a, b, c, d = self.plane_model
            print(f"\n平面方程：{a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            print(f"平面内点数量：{len(self.inlier_cloud.points)}")
            print(f"平面外点数量：{len(self.outlier_cloud.points)}")
            return True
        except Exception as e:
            print(f"平面分割失败：{str(e)}")
            return False

    def cluster_dbscan(self, eps: float = 10.0, min_points: int = 10) -> bool:
        """
        DBSCAN聚类：对平面内点进行聚类分析
        Args:
            eps: 邻域半径（默认：10.0）
            min_points: 形成聚类的最小点数（默认：10）
        Returns: 聚类成功返回True，失败返回False
        """
        if self.inlier_cloud is None:
            print("请先调用segment_plane()完成平面分割")
            return False

        try:
            # 执行DBSCAN聚类（开启调试日志）
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
                self.cluster_labels = np.array(
                    self.inlier_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
                )
            # 统计聚类信息
            max_label = self.cluster_labels.max()
            unique_elements, counts = np.unique(self.cluster_labels, return_counts=True)
            print(f"\n聚类结果：共检测到 {max_label + 1} 个有效聚类（排除噪声点）")
            print(f"各聚类点数：{dict(zip(unique_elements, counts))}")
            return True
        except Exception as e:
            print(f"DBSCAN聚类失败：{str(e)}")
            return False

    def select_most_frequent_cluster(self) -> bool:
        """
        筛选出现次数最多的聚类（核心目标聚类）
        Returns: 筛选成功返回True，失败返回False
        """
        if self.cluster_labels is None or self.inlier_cloud is None:
            print("请先调用cluster_dbscan()完成聚类")
            return False

        try:
            # 获取高频聚类的索引
            target_indices = self.get_most_frequent_number_indices(self.cluster_labels)
            # 提取目标聚类点云
            self.target_cluster_cloud = self.inlier_cloud.select_by_index(target_indices)
            print(f"\n筛选出最大聚类：点数 = {len(self.target_cluster_cloud.points)}")
            return True
        except Exception as e:
            print(f"聚类筛选失败：{str(e)}")
            return False

    def visualize(self, target: str = "target_cluster") -> None:
        """
        可视化点云结果
        Args:
            target: 可视化目标，可选值：
                - "inlier": 平面内点
                - "all_clusters": 所有聚类（带颜色区分）
                - "target_cluster": 目标聚类（默认）
        """
        valid_targets = ["inlier", "all_clusters", "target_cluster"]
        if target not in valid_targets:
            print(f"无效的可视化目标，可选值：{valid_targets}")
            return

        try:
            if target == "inlier":
                if self.inlier_cloud is None:
                    print("无平面内点数据可可视化")
                    return
                o3d.visualization.draw_geometries([self.inlier_cloud], window_name="平面内点云")

            elif target == "all_clusters":
                if self.inlier_cloud is None or self.cluster_labels is None:
                    print("无聚类数据可可视化")
                    return
                # 为不同聚类分配颜色
                max_label = self.cluster_labels.max()
                colors = plt.get_cmap("tab20")(self.cluster_labels / (max_label if max_label > 0 else 1))
                colors[self.cluster_labels < 0] = 0  # 噪声点设为黑色
                self.inlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
                o3d.visualization.draw_geometries([self.inlier_cloud], window_name="所有聚类点云")

            elif target == "target_cluster":
                if self.target_cluster_cloud is None:
                    print("无目标聚类数据可可视化")
                    return
                o3d.visualization.draw_geometries(
                    [self.target_cluster_cloud],
                    window_name="目标聚类点云",
                    zoom=0.455,
                    front=[-0.4999, -0.1659, -0.8499],
                    lookat=[2.1813, 2.0619, 2.0999],
                    up=[0.1204, -0.9852, 0.1215]
                )
        except Exception as e:
            print(f"可视化失败：{str(e)}")

    def save_results(self, save_inlier: bool = True, save_target_cluster: bool = True, save_txt: bool = False) -> None:
        """
        保存处理结果（支持PLY格式和可选的TXT格式）
        Args:
            save_inlier: 是否保存平面内点（默认：True）
            save_target_cluster: 是否保存目标聚类点（默认：True）
            save_txt: 是否同时保存TXT格式（默认：False）
        """
        try:
            # 保存平面内点
            if save_inlier and self.inlier_cloud is not None:
                # PLY格式（主格式）
                inlier_ply_path = os.path.join(self.output_dir, "output_inlier.ply")
                if not self.inlier_cloud.has_colors():
                    self.inlier_cloud.paint_uniform_color([1.0, 1.0, 1.0])
                o3d.io.write_point_cloud(inlier_ply_path, self.inlier_cloud)
                print(f"\n平面内点(PLY)已保存至：{inlier_ply_path}")

                # 可选：保存TXT格式
                if save_txt:
                    inlier_txt_path = os.path.join(self.output_dir, "output_inlier.txt")
                    np.savetxt(inlier_txt_path, np.asarray(self.inlier_cloud.points))
                    print(f"平面内点(TXT)已保存至：{inlier_txt_path}")

            # 保存目标聚类点
            if save_target_cluster and self.target_cluster_cloud is not None:
                # PLY格式（主格式）
                target_ply_path = os.path.join(self.output_dir, "output_target_cluster.ply")
                if not self.target_cluster_cloud.has_colors():
                    self.target_cluster_cloud.paint_uniform_color([1.0, 0.0, 0.0])
                o3d.io.write_point_cloud(target_ply_path, self.target_cluster_cloud)
                print(f"目标聚类点(PLY)已保存至：{target_ply_path}")

                # 可选：保存TXT格式
                if save_txt:
                    target_txt_path = os.path.join(self.output_dir, "output_target_cluster.txt")
                    np.savetxt(target_txt_path, np.asarray(self.target_cluster_cloud.points))
                    print(f"目标聚类点(TXT)已保存至：{target_txt_path}")
        except Exception as e:
            print(f"结果保存失败：{str(e)}")

    def run_pipeline(self,
                    plane_params: Optional[dict] = None,
                    cluster_params: Optional[dict] = None,
                    visualize_steps: bool = True,
                    save_results: bool = True) -> bool:
        """
        执行完整处理流程：读取点云 → 平面分割 → 聚类 → 筛选目标聚类 → 可视化 → 保存结果
        Args:
            plane_params: 平面分割参数（字典），默认使用类默认值
            cluster_params: DBSCAN聚类参数（字典），默认使用类默认值
            visualize_steps: 是否可视化各步骤结果（默认：True）
            save_results: 是否保存结果文件（默认：True）
        Returns: 流程执行成功返回True，失败返回False
        """
        # 使用默认参数（若未传入）
        plane_params = plane_params or {}
        cluster_params = cluster_params or {}

        print("=" * 50)
        print("开始点云处理流程...")
        print("=" * 50)

        # 步骤1：读取点云
        if not self.load_point_cloud():
            return False

        # 步骤2：平面分割
        if not self.segment_plane(**plane_params):
            return False
        if visualize_steps:
            self.visualize("inlier")

        # 步骤3：DBSCAN聚类
        if not self.cluster_dbscan(**cluster_params):
            return False
        if visualize_steps:
            self.visualize("all_clusters")

        # 步骤4：筛选目标聚类
        if not self.select_most_frequent_cluster():
            return False
        if visualize_steps:
            self.visualize("target_cluster")

        # 步骤5：保存结果
        if save_results:
            self.save_results()

        print("\n" + "=" * 50)
        print("点云处理流程完成！")
        print("=" * 50)
        return True


# ---------------------- 示例：使用封装类 ----------------------
if __name__ == "__main__":
    # 1. 配置参数
    INPUT_PCD_PATH = "./3-clean.ply"  # 输入点云路径
    OUTPUT_DIR = "./point_cloud_deal"                    # 输出目录

    # 自定义参数（可选，不传入则使用默认值）
    PLANE_PARAMS = {
        "distance_threshold": 3.0,
        "ransac_n": 3,
        "num_iterations": 1000
    }

    CLUSTER_PARAMS = {
        "eps": 10.0,
        "min_points": 10
    }

    # 2. 创建处理器实例
    processor = PointCloudProcessor(
        input_pcd_path=INPUT_PCD_PATH,
        output_dir=OUTPUT_DIR
    )

    # 3. 执行完整流程（可控制是否可视化、是否保存）
    processor.run_pipeline(
        plane_params=PLANE_PARAMS,
        cluster_params=CLUSTER_PARAMS,
        visualize_steps=True,  # True=显示可视化窗口，False=不显示
        save_results=True      # True=保存txt文件，False=不保存
    )

    # ---------------------- 灵活使用示例（单独调用步骤）----------------------
    # # 若需分步执行（如调试），可单独调用方法：
    # processor.load_point_cloud()
    # processor.segment_plane(distance_threshold=2.5)  # 自定义平面分割阈值
    # processor.visualize("inlier")
    # processor.cluster_dbscan(eps=8.0)  # 自定义聚类半径
    # processor.select_most_frequent_cluster()
    # processor.save_results(save_inlier=False)  # 只保存目标聚类