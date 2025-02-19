import open3d as o3d
import copy
import numpy as np

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([1, 0.706, 0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    # 点云合并
    # 获取第一个点云的点坐标
    source.transform(transformation)
    points1 = np.asarray(source.points)
    # 获取第二个点云的点坐标
    points2 = np.asarray(target.points)
    # 合并点坐标
    merged_points = np.vstack((points1, points2))

    # 创建新的点云对象
    merged_pcd = o3d.geometry.PointCloud()
    # 设置合并后的点坐标
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    return merged_pcd
source_ = o3d.io.read_point_cloud("292.pcd")
source = source_.voxel_down_sample(voxel_size=5)
target1_ = o3d.io.read_point_cloud("484.pcd")
target1 = target1_.voxel_down_sample(voxel_size=5)
target2_ = o3d.io.read_point_cloud("676.pcd")
target2 = target2_.voxel_down_sample(voxel_size=5)
threshold = 15
trans_init = np.asarray([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0.0, 0.0, 0.0, 1.0]])
# draw_registration_result(source, target1, trans_init)
reg_p2p = o3d.pipelines.registration.registration_icp(source, target1, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 200000))

# draw_registration_result(source, target1, trans_init)
reg_p2p_ = o3d.pipelines.registration.registration_icp(target1, target2, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 200000))

print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
point1_2 = draw_registration_result(source, target1, reg_p2p.transformation)
point1_2_3 = draw_registration_result(point1_2, target2, reg_p2p_.transformation)