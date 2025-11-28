import cv2
import numpy as np
import os
from typing import Optional, Tuple

class InfraredDotDetector:
    def __init__(self, grid_size: Tuple[int, int] = (5, 5)):
        """
        红外光点阵列圆心检测器
        Args:
            grid_size: 光点阵列的行列数（默认5x5，根据实际场景调整）
        """
        self.grid_size = grid_size  # (rows, cols) 光点阵列尺寸
        self.detector = self._create_blob_detector()  # 初始化Blob检测器

    def _create_blob_detector(self) -> cv2.SimpleBlobDetector:
        """创建适配红外光点的Blob检测器（核心参数优化）"""
        params = cv2.SimpleBlobDetector_Params()

        # 1. 面积过滤（适配小尺寸红外光点）
        params.filterByArea = True
        params.minArea = 20    # 最小面积（降低阈值适配小光点）
        params.maxArea = 2000  # 最大面积（过滤大噪声块）

        # 2. 圆度过滤（放宽要求，适配模糊光点）
        params.filterByCircularity = True
        params.minCircularity = 0.4  # 圆度≥0.4即可（原0.7，适配模糊场景）

        # 3. 惯性比过滤（允许不规则亮斑）
        params.filterByInertia = True
        params.minInertiaRatio = 0.3  # 原0.5，放宽形状限制

        # 4. 关闭凸度过滤（避免过滤边缘不规整的光点）
        params.filterByConvexity = False

        # 5. 亮度过滤（可选，根据红外图实际亮度调整）
        # params.filterByBrightness = True
        # params.minBrightness = 150  # 仅保留亮于150的区域（暗背景亮斑场景）

        # 创建并返回检测器
        return cv2.SimpleBlobDetector_create(params)

    def detect_dot_centers(self, img_path, choose_destroy=True):
        """
        核心优化：检测红外图中的亮光点阵列中心
        适配特点：暗背景、亮光点、小尺寸、可能模糊
        """
        # 读取红外图像（可能是单通道灰度图）
        img = cv2.imread(img_path)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img  # 已是灰度图

        # -------------------------- 关键优化：红外光点检测参数 --------------------------
        params = cv2.SimpleBlobDetector_Params()

        # 1. 过滤基于亮度（红外光点是亮的，保留亮斑）
        # params.filterByBrightness = True
        # params.minBrightness = 150  # 只保留亮度高于150的区域（根据红外图亮度调整）

        # 2. 过滤基于面积（适配小光点，缩小最小面积阈值）
        params.filterByArea = True
        params.minArea = 20  # 红外光点可能很小，降低最小面积（原代码是50）
        params.maxArea = 2000  # 限制最大面积，过滤噪声块

        # 3. 过滤基于圆度（红外光点可能模糊，降低圆度要求）
        params.filterByCircularity = True
        params.minCircularity = 0.4  # 原代码是0.7，适配模糊光点（圆度≥0.4即可）

        # 4. 过滤基于惯性比（放宽要求，适配不规则亮斑）
        params.filterByInertia = True
        params.minInertiaRatio = 0.3  # 原代码是0.5，允许更多形状的光点

        # 5. 关闭基于凸度（红外光点边缘可能不凸，避免误过滤）
        params.filterByConvexity = False

        # -------------------------- 关键优化：反转图像（可选，根据光点类型） --------------------------
        # 如果光点是“暗背景中的亮斑”，无需反转；如果是“亮背景中的暗斑”，则启用反转
        # gray = cv2.bitwise_not(gray)  # 按需注释/取消注释

        # 创建Blob检测器（适配红外光点）
        detector = cv2.SimpleBlobDetector_create(params)

        # -------------------------- 关键优化：光点阵列检测标志 --------------------------
        # CALIB_CB_CLUSTERING：允许光点有轻微偏移，适合不规则阵列
        # CALIB_CB_SYMMETRIC_GRID：对称阵列（你的红外图是5x5对称光点）
        ret, centers = cv2.findCirclesGrid(
            gray, self.grid_size,
            blobDetector=detector,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
        )

        if ret:
            # # 绘制检测到的光点中心和连线（便于验证）
            # if len(img.shape) == 2:
            #     img_with_dots = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # else:
            #     img_with_dots = img.copy()
            #
            # img_with_dots = cv2.drawChessboardCorners(
            #     img_with_dots, self.grid_size, centers, ret
            # )
            # img_name = os.path.basename(img_path)  # 只取文件名（如 "ir_calib_00.jpg"）
            # window_title = f"Infrared Detection: {img_name}"  # 合法标题格式
            # # 显示检测结果（300ms自动关闭）
            # cv2.imshow(window_title, img_with_dots)
            # if choose_destroy:
            #     cv2.waitKey(500)  # 延长显示时间，便于观察
            #     # cv2.destroyWindow("Infrared Dot Grid Detection")
            #     return centers.astype(np.float32),img
            # else:
            return centers.astype(np.float32),img
        else:
            print(f"警告：图像 {img_path} 未检测到光点阵列！")
            return None,img

    def visualize_and_save(self, img_path: str, output_dir: str = "./dot_detection_results") -> None:
        """
        检测圆心并生成可视化PNG结果（标注圆心+坐标）
        Args:
            img_path: 输入图像路径
            output_dir: 输出结果保存目录（默认：./dot_detection_results）
        """
        # 1. 检测圆心
        centers, img = self.detect_dot_centers(img_path)
        if img is None:
            return

        # 2. 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        if len(img.shape) == 2:
            img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_vis = img.copy()

        img_vis = cv2.drawChessboardCorners(
            img_vis, self.grid_size, centers, True
        )
        centers = centers.squeeze()

        if centers is not None:
            # 绘制圆心（红色实心圆，半径3，抗锯齿）
            i = 0
            for (x, y) in centers:
                i = i + 1
                print(f"第{i}个坐标为：{x},{y}")

                # 绘制坐标文本（蓝色字体，避免遮挡圆心）
                coord_text = f"{i}"
                cv2.putText(
                    img_vis, coord_text, (int(x) + 4, int(y) - 4),  # 文本位置在圆心右下方
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA
                )
        else:
            # 未检测到光点时添加提示文本
            cv2.putText(
                img_vis, "No dot centers detected", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
            )

        # 4. 保存结果PNG
        img_name = os.path.basename(img_path)
        output_name = os.path.splitext(img_name)[0] + "_centers_detected.png"
        output_path = os.path.join(output_dir, output_name)

        # 无损保存PNG
        cv2.imwrite(output_path, img_vis, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"可视化结果已保存至：{output_path}")

        # 可选：显示结果（5秒后自动关闭）
        cv2.imshow("Dot Centers Detection Result", img_vis)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

# ---------------------- 示例调用 ----------------------
if __name__ == "__main__":
    # 1. 配置参数
    INPUT_IMAGE_PATH = "./ir_image/20250314_220414(14).jpg"  # 替换为你的红外图像路径
    OUTPUT_DIR = "./dot_detection_results"         # 输出目录
    GRID_SIZE = (5, 5)                             # 光点阵列尺寸（行列数，根据实际调整）

    # 2. 创建检测器实例
    detector = InfraredDotDetector(grid_size=GRID_SIZE)

    # 3. 执行检测+可视化+保存
    detector.visualize_and_save(
        img_path=INPUT_IMAGE_PATH,
        output_dir=OUTPUT_DIR
    )

    # ---------------------- 批量处理示例（可选）----------------------
    # input_folder = "./infrared_images"  # 红外图像文件夹路径
    # for img_file in os.listdir(input_folder):
    #     if img_file.lower().endswith((".jpg", ".png", ".bmp", ".tiff")):
    #         img_path = os.path.join(input_folder, img_file)
    #         detector.visualize_and_save(img_path, output_dir)