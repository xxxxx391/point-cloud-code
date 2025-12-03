"""
可见光与红外逐像素配准与融合（输入：visible.jpg, ir.jpg, pointcloud.ply）
修改功能：将可见光纹理映射到红外图像上

说明：
- 本脚本以点云（PLY）代替结构光深度图。点云应在“结构光”相机坐标系下（即点坐标与外参的参考系一致）。
- 外参从txt文件读取（4×4矩阵），内参需在脚本顶部填入
- 外参文件格式：每行4个数字，共4行，以空格或逗号分隔

输入：
- visible.jpg (可见光 RGB)
- ir.jpg      (红外单通道或RGB，脚本会处理)
- cloud.ply   (点云文件，支持 ASCII 和二进制 PLY)
- extrinsic_s2vis.txt (结构光到可见光的4×4外参矩阵)
- extrinsic_s2ir.txt  (结构光到红外的4×4外参矩阵)

输出：
- warped_vis_to_ir.png    （将可见光纹理映射到红外视角）
- fused_ir_vis.png        （融合结果：红外图像 + 可见光纹理）

注意：
- 点云较大时，逐点循环会较慢；可替换为基于 NumPy 的向量化分组或用加速库（Numba/Open3D/CUDA）来提速。
- 外参txt文件需确保是纯数字，每行4个值，共4行，无多余字符。
"""

import os
import numpy as np
import cv2
import open3d as o3d  # 导入Open3D库

# ------------------------ 辅助函数（保持不变） ------------------------

def load_extrinsic_from_txt(file_path):
    """
    从txt文件加载4×4外参矩阵
    支持格式：每行4个数字，以空格或逗号分隔，共4行
    返回：4×4 numpy数组（float32）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"外参文件不存在：{file_path}")

    # 读取所有行，过滤空行和注释行
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    # 检查行数是否为4
    if len(lines) != 4:
        raise ValueError(f"外参文件需包含4行数据，实际读取到{len(lines)}行")

    extrinsic = []
    for i, line in enumerate(lines):
        # 支持空格或逗号分隔
        if ',' in line:
            values = line.split(',')
        else:
            values = line.split()

        # 转换为浮点数并过滤空字符串
        values = [float(v.strip()) for v in values if v.strip()]

        # 检查每行是否有4个数值
        if len(values) != 4:
            raise ValueError(f"第{i+1}行需包含4个数值，实际读取到{len(values)}个")

        extrinsic.append(values)

    # 转换为numpy数组并确保形状为4×4
    extrinsic_mat = np.array(extrinsic, dtype=np.float32)
    if extrinsic_mat.shape != (4, 4):
        raise ValueError(f"外参矩阵形状错误，应为(4,4)，实际为{extrinsic_mat.shape}")

    return extrinsic_mat

def split_extrinsic(extrinsic_mat):
    """
    从4×4外参矩阵中拆分旋转矩阵R（3×3）和平移向量t（3×1）
    外参矩阵格式：[R t; 0 0 0 1]
    返回：R (3×3), t (3×1)
    """
    if extrinsic_mat.shape != (4, 4):
        raise ValueError(f"外参矩阵必须是4×4，实际为{extrinsic_mat.shape}")

    R = extrinsic_mat[:3, :3]
    t = extrinsic_mat[:3, 3:4]  # 保持为3×1矩阵，适配后续矩阵运算
    return R, t

def load_ply_vertices_with_open3d(path):
    """
    使用Open3D读取PLY文件，返回点云坐标和额外属性
    支持ASCII和二进制格式的PLY文件
    返回: points (N,3), extra_props (dict of arrays)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"点云文件不存在：{path}")

    # 读取点云
    pcd = o3d.io.read_point_cloud(path)

    # 检查是否成功读取点云
    if not pcd.has_points():
        raise ValueError("读取的点云不包含任何点")

    # 获取点坐标 (N, 3)
    points = np.asarray(pcd.points, dtype=np.float32)

    # 收集额外属性（颜色、强度等）
    extra_props = {}

    # 如果有点云颜色（RGB）
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)  # 颜色值在[0,1]之间
        extra_props['rgb'] = colors
        # 也可以转换为0-255的整数格式
        extra_props['rgb_255'] = (colors * 255).astype(np.uint8)

    return points, extra_props

def bilinear_sample_single_channel(img, x, y):
    H, W = img.shape
    if x < 0 or x >= W-1 or y < 0 or y >= H-1:
        return 0.0
    x0 = int(np.floor(x)); x1 = x0 + 1
    y0 = int(np.floor(y)); y1 = y0 + 1
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def bilinear_sample_color(img, x, y):
    if img.ndim == 2:
        v = bilinear_sample_single_channel(img, x, y)
        return np.array([v, v, v], dtype=np.float32)
    H, W, C = img.shape
    if x < 0 or x >= W-1 or y < 0 or y >= H-1:
        return np.zeros((C,), dtype=np.float32)
    x0 = int(np.floor(x)); x1 = x0 + 1
    y0 = int(np.floor(y)); y1 = y0 + 1
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    Ia = img[y0, x0].astype(np.float32)
    Ib = img[y0, x1].astype(np.float32)
    Ic = img[y1, x0].astype(np.float32)
    Id = img[y1, x1].astype(np.float32)
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

# ------------------------ 核心修改：可见光纹理映射到红外图像 ------------------------

def warp_visible_to_ir_from_points(points_s, K_vis, K_ir, R_s2vis, t_s2vis, R_s2ir, t_s2ir,
                                   vis_img, ir_img, zbuffer_eps=1e-6):
    """
    使用点云（位于结构光相机坐标系）把可见光纹理映射到红外图像像素上。
    核心逻辑：
    1. 将结构光坐标系的点转换到可见光相机坐标系，投影得到可见光图像上的像素坐标
    2. 从可见光图像上采样纹理颜色
    3. 将同一个3D点转换到红外相机坐标系，投影得到红外图像上的像素坐标
    4. 将采样到的可见光纹理颜色赋值到红外图像的对应像素位置
    5. 使用深度缓冲保证近处的点覆盖远处的点

    points_s: (N,3) 结构光坐标系下的点云
    K_vis: 可见光相机内参
    K_ir: 红外相机内参
    R_s2vis, t_s2vis: 结构光到可见光的旋转矩阵和平移向量
    R_s2ir, t_s2ir: 结构光到红外的旋转矩阵和平移向量
    vis_img: 可见光图像
    ir_img: 红外图像
    返回: warped_vis (与 ir_img 同大小), valid_mask, depth_buffer
    """
    H_ir, W_ir = ir_img.shape[:2]  # 输出图像大小与红外图像一致
    H_vis, W_vis = vis_img.shape[:2]

    N = points_s.shape[0]
    # 1. 转换3D点到可见光相机坐标系，计算可见光图像上的投影坐标
    X_vis = (R_s2vis @ points_s.T) + t_s2vis  # 3 x N
    uvw_vis = K_vis @ X_vis
    u_vis = uvw_vis[0,:] / (uvw_vis[2,:] + zbuffer_eps)  # 归一化坐标
    v_vis = uvw_vis[1,:] / (uvw_vis[2,:] + zbuffer_eps)
    depth_vis = X_vis[2,:]  # 可见光相机坐标系下的深度

    # 2. 转换同一个3D点到红外相机坐标系，计算红外图像上的投影坐标
    X_ir = (R_s2ir @ points_s.T) + t_s2ir  # 3 x N
    uvw_ir = K_ir @ X_ir
    u_ir = uvw_ir[0,:] / (uvw_ir[2,:] + zbuffer_eps)  # 归一化坐标
    v_ir = uvw_ir[1,:] / (uvw_ir[2,:] + zbuffer_eps)
    depth_ir = X_ir[2,:]  # 红外相机坐标系下的深度

    # 结果缓冲初始化
    warped_vis = np.zeros_like(ir_img, dtype=np.float32)  # 输出与红外图像同大小
    depth_buffer = np.full((H_ir, W_ir), np.inf, dtype=np.float32)  # 深度缓冲，初始为无穷大
    valid_mask = np.zeros((H_ir, W_ir), dtype=np.bool_)  # 标记有效映射的像素

    # 逐点处理
    for i in range(N):
        # 过滤无效深度（深度必须为正且有限）
        z_ir = depth_ir[i]
        if z_ir <= 0 or not np.isfinite(z_ir):
            continue

        # 获取红外图像上的投影坐标
        u = u_ir[i]
        v = v_ir[i]

        # 检查坐标是否在红外图像范围内
        # if not (0 <= u < W_ir and 0 <= v < H_ir):
        #     continue

        # 转换为整数像素坐标
        x_ir = int(round(u))
        y_ir = int(round(v))

        if not (0 <= x_ir < W_ir and 0 <= y_ir < H_ir):
            continue
        # 深度缓冲测试：只有当前点比已存储的点更近时才更新
        if z_ir < depth_buffer[y_ir, x_ir]:
            # 从可见光图像采样纹理颜色
            u_vis_i = u_vis[i]
            v_vis_i = v_vis[i]

            # 检查可见光采样坐标是否有效
            if not (0 <= u_vis_i < W_vis-1 and 0 <= v_vis_i < H_vis-1):
                continue

            # 双线性采样获取可见光颜色（保证采样质量）
            sampled_vis_color = bilinear_sample_color(vis_img, u_vis_i, v_vis_i)

            # 更新结果
            warped_vis[y_ir, x_ir] = sampled_vis_color
            depth_buffer[y_ir, x_ir] = z_ir
            valid_mask[y_ir, x_ir] = True

    # 通道数统一：如果红外是单通道，可见光纹理也转为单通道
    if ir_img.ndim == 2 and warped_vis.ndim == 3:
        warped_vis = cv2.cvtColor(warped_vis.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    return warped_vis, valid_mask, depth_buffer

# ------------------------ 融合函数（调整输入顺序，保持逻辑不变） ------------------------

def fuse_ir_and_visible(ir_img, warped_vis, valid_mask, method='weighted', **kwargs):
    """
    融合红外图像和映射后的可见光纹理
    输入：
    - ir_img: 原始红外图像（基准图像）
    - warped_vis: 映射到红外视角的可见光纹理
    - valid_mask: 有效映射区域掩码
    - method: 融合方法（weighted/replace_dark/overlay_vis）
    返回：融合后的图像
    """
    ir = ir_img.astype(np.float32)
    vis = warped_vis.astype(np.float32)
    H, W = valid_mask.shape

    # 通道数统一
    if vis.ndim == 2:
        vis = np.repeat(vis[:,:,None], 3, axis=2)
    if ir.ndim == 2:
        ir = np.repeat(ir[:,:,None], 3, axis=2)

    fused = ir.copy()  # 以红外图像为基准

    if method == 'weighted':
        # 加权融合：可通过alpha控制可见光纹理的权重
        alpha = kwargs.get('alpha', 0.4)  # 可见光纹理的权重（默认0.4，可调整）
        if 'vis_weight_map' in kwargs:
            wmap = kwargs['vis_weight_map']
            wmap = np.clip(wmap, 0.0, 1.0)
            if wmap.shape != (H, W):
                raise ValueError('vis_weight_map 大小应为 HxW')
            wmap3 = np.repeat(wmap[:,:,None], 3, axis=2)
            mask3 = np.repeat(valid_mask[:,:,None], 3, axis=2)
            fused[mask3] = (1.0 - wmap3[mask3]) * ir[mask3] + wmap3[mask3] * vis[mask3]
        else:
            mask3 = np.repeat(valid_mask[:,:,None], 3, axis=2)
            fused[mask3] = (1.0 - alpha) * ir[mask3] + alpha * vis[mask3]

    elif method == 'replace_dark':
        # 替换红外图像中的暗区域：红外图像较暗的地方用可见光纹理补充
        gray_ir = cv2.cvtColor(ir.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        ir_norm = gray_ir / 255.0
        darkness = 1.0 - ir_norm
        thresh = kwargs.get('dark_thresh', 0.35)
        weight = np.clip((darkness - thresh) / (1.0 - thresh), 0.0, 1.0)
        w3 = np.repeat(weight[:,:,None], 3, axis=2)
        mask3 = np.repeat(valid_mask[:,:,None], 3, axis=2)
        fused[mask3] = (1.0 - w3[mask3]) * ir[mask3] + w3[mask3] * vis[mask3]

    elif method == 'overlay_vis':
        # 叠加可见光纹理（保留红外图像的轮廓，叠加可见光的细节）
        alpha = kwargs.get('alpha', 0.5)
        mask3 = np.repeat(valid_mask[:,:,None], 3, axis=2)
        fused[mask3] = (1.0 - alpha) * ir[mask3] + alpha * vis[mask3]
    else:
        raise ValueError('未知融合方法')

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused

# ------------------------ 示例调用（修改映射和融合函数调用） ------------------------
if __name__ == '__main__':
    # --- 在这里填入你的内参（需根据实际相机标定结果修改） ---
    K_vis = np.array([[1763.6, 0.0, 987.4650],
                      [0.0, 1765.7, 530.6634],
                      [0.0,   0.0,   1.0]], dtype=np.float32)

    K_ir = np.array([[577.6673, 0.0, 188.6225],
                     [0.0, 577.4241, 148.4147],
                     [0.0,   0.0,   1.0]], dtype=np.float32)

    # --- 外参文件路径：替换为你的外参txt文件路径 ---
    extrinsic_s2vis_path = 'transform_matrix_kejian.txt'  # 结构光→可见光 外参
    extrinsic_s2ir_path = './ir_visible_point/save_red2point_1203/transform_matrix_red2point.txt'    # 结构光→红外 外参

    # 加载并拆分外参
    print('加载外参矩阵...')
    extrinsic_s2vis = load_extrinsic_from_txt(extrinsic_s2vis_path)
    extrinsic_s2ir = load_extrinsic_from_txt(extrinsic_s2ir_path)

    R_s2vis, t_s2vis = split_extrinsic(extrinsic_s2vis)
    R_s2ir, t_s2ir = split_extrinsic(extrinsic_s2ir)

    print('结构光→可见光 外参拆分完成：')
    print(f'旋转矩阵 R:\n{R_s2vis}')
    print(f'平移向量 t:\n{t_s2vis}')

    # --- 其他文件路径：替换为你的文件 ---
    vis_path = './ir_visible_point/20251202_213632V.jpg'
    ir_path = './ir_visible_point/20251202_213632.jpg'
    ply_path = './ir_visible_point/1-Cloud.ply'

    # 读取图像
    vis = cv2.imread(vis_path, cv2.IMREAD_COLOR)
    if vis is None:
        raise FileNotFoundError(f'找不到可见光图像: {vis_path}')
    ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    if ir is None:
        raise FileNotFoundError(f'找不到红外图像: {ir_path}')

    # 预处理红外图像（保持与原代码一致）
    if ir.ndim == 2:
        ir_vis = np.repeat(ir[:,:,None], 3, axis=2)
    else:
        ir_vis = ir.copy()

    # 读取点云（使用Open3D）
    print('使用Open3D读取点云...')
    points, extras = load_ply_vertices_with_open3d(ply_path)
    print(f'加载点云: {points.shape[0]} 点')

    # 打印额外属性信息（可选）
    if extras:
        print(f'点云包含额外属性: {list(extras.keys())}')
        for key, value in extras.items():
            print(f'  {key}: 形状 {value.shape}')

    # ------------------------ 核心修改：调用新的映射函数 ------------------------
    # 将可见光纹理映射到红外图像
    print('正在将可见光纹理映射到红外图像...')
    warped_vis, valid_mask, depth_buf = warp_visible_to_ir_from_points(points, K_vis, K_ir,
                                                                      R_s2vis, t_s2vis, R_s2ir, t_s2ir,
                                                                      vis, ir_vis)

    # ------------------------ 融合与保存 ------------------------
    # 转换为uint8格式用于保存
    warped_vis_ir = warped_vis.copy().astype(np.uint8)

    # 融合：红外图像 + 映射后的可见光纹理（使用加权融合，可调整alpha值）
    fused = fuse_ir_and_visible(ir_vis, warped_vis_ir, valid_mask, method='weighted', alpha=0.4)

    # 保存结果
    cv2.imwrite('warped_vis_to_ir.png', warped_vis_ir)
    cv2.imwrite('fused_ir_vis.png', fused)

    # 可选：保存有效掩码（方便调试）
    mask_vis = (valid_mask * 255).astype(np.uint8)
    cv2.imwrite('valid_mask.png', mask_vis)

    print('完成：已生成以下文件：')
    print('  - warped_vis_to_ir.png: 可见光纹理映射到红外视角的结果')
    print('  - fused_ir_vis.png: 红外图像与可见光纹理的融合结果')
    print('  - valid_mask.png: 有效映射区域的掩码（白色为有效区域）')