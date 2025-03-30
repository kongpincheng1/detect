import warnings
warnings.simplefilter('ignore', category=FutureWarning)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
from pathlib import Path
import math
from ultralytics import YOLO

class YOLOv5ROS2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')

        # 订阅彩色图像 (需要你确保话题名字正确)
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        # 订阅深度图像
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )

        # 发布目标真实世界坐标
        self.publisher = self.create_publisher(Point, '/target_position', 10)
        self.centerHeight_Pub = self.create_publisher(Float32, '/current_height', 10)

        # -------------------- 加载 YOLOv5 模型 ------------------------
        weights_path = '/home/cqu/test_ws/best.pt'
        self.model = torch.hub.load(
            repo_or_dir='/home/cqu/yolov5',
            model='custom',
            path=weights_path,
            source='local',
            force_reload=True
        )
        # self.model = YOLO("/home/cqu/test_ws/best.pt") 
        self.model.eval()

        # OpenCV 桥
        self.bridge = CvBridge()

        # 缓存最近的彩色图、深度图
        self.color_image = None
        self.depth_image = None

        # 相机内参（可根据实际标定来改）
        self.fx = 605.7783203125
        self.fy = 605.474609375
        self.cx = 326.34991455078125
        self.cy = 242.88038635253906

        self.get_logger().info('YOLOv5 ROS 2 Node Initialized!')

    def image_callback(self, msg):
        """彩色图像回调"""
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.depth_image is not None:
                self.process_images()
        except Exception as e:
            self.get_logger().error(f"Failed to process color image: {e}")

    def depth_callback(self, msg):
        """深度图像回调"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if self.color_image is not None:
                self.process_images()
        except Exception as e:
            self.get_logger().error(f"Failed to process depth image: {e}")

    def process_images(self):
        """同时处理彩色图 + 深度图"""

        # ========== 1) 发布画面中心的深度 ==========
        depth_center = self.depth_image[320, 240] * 0.001  # mm -> m (假设深度图是 mm)
        center_height = Float32()
        center_height.data = depth_center if depth_center > 0 else 0.0
        self.centerHeight_Pub.publish(center_height)

        # ========== 2) YOLOv5 目标检测 ==========
        results = self.detect_objects(self.color_image)

        # 【先拷贝一份图像做可视化标注】
        rgb_copy = self.color_image.copy()

        # ========== 3) 遍历检测到的目标框 ==========
        for result in results:
            x1, y1, x2, y2, conf, cls = result

            # 取中心点
            cx_pixel = int((x1 + x2) / 2)
            cy_pixel = int((y1 + y2) / 2)

            # (A) 获取ROI中位深度 => 估计圆筒真实直径 (可选)
            median_depth = self.get_roi_median_depth(x1, y1, x2, y2)
            if median_depth <= 0:
                # 如果ROI里无效深度，就退化到用中心像素深度
                fallback_depth = self.depth_image[cy_pixel, cx_pixel] * 0.001
                median_depth = fallback_depth if fallback_depth > 0 else 0

            # (B) 计算圆筒在图像中的宽度像素 => 估计真实直径
            bbox_width_px = (x2 - x1)
            real_diameter = 0
            if median_depth > 0:
                real_diameter = (bbox_width_px / self.fx) * median_depth

            # (C) 根据阈值分类大小 => 返回 'S' / 'M' / 'B'
            size_label = self.classify_cylinder_size(real_diameter)

            # (D) 获取中心点的深度
            depth_at_center = self.depth_image[cy_pixel, cx_pixel] * 0.001
            if depth_at_center <= 0:
                depth_at_center = median_depth

            # (E) 得到世界坐标
            world_x, world_y, _ = self.pixel_to_world(cx_pixel, cy_pixel, depth_at_center)

            # (F) 发布到 /target_position (复用 z 字段放大小 1/2/3)
            point_msg = Point()
            point_msg.x = world_x
            point_msg.y = world_y
            if size_label == 'S':
                point_msg.z = 1.0
            elif size_label == 'M':
                point_msg.z = 2.0
            else:
                point_msg.z = 3.0
            self.publisher.publish(point_msg)

            # ============= 在 rgb_copy 上画框 & 标注 =============
            cv2.rectangle(rgb_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            label_text = f"{size_label} {conf:.2f},{real_diameter:.5f}"
            cv2.putText(rgb_copy, label_text, (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ========== 4) 用 OpenCV 显示检测结果 ==========
        cv2.imshow("Detection", rgb_copy)
        cv2.waitKey(1)

    @torch.no_grad
    def detect_objects(self, image):
        """调用YOLOv5进行推理"""
        results = self.model(image, size=320)
        detections = results.xyxy[0].cpu().numpy()  # [N, 6], (x1,y1,x2,y2,conf,cls)
        # 过滤低置信度
        detections = detections[detections[:, 4] > 0.4]
        return detections

    def pixel_to_world(self, u, v, depth):
        """像素坐标 + 深度 => 真实世界坐标 (X, Y, Z)"""
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z

    def get_roi_median_depth(self, x1, y1, x2, y2):
        """对 bounding box 区域在深度图中取中位数"""
        h, w = self.depth_image.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w - 1, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h - 1, int(y2)))
        roi = self.depth_image[y1:y2, x1:x2].flatten()
        roi_valid = roi[(roi > 0) & (roi < 10000)]  # 去掉 0 和异常值
        if len(roi_valid) == 0:
            return -1.0
        med_mm = np.median(roi_valid)
        return med_mm * 0.001  # 转米

    def classify_cylinder_size(self, diameter_m):
        """
        根据直径阈值分类: 返回 'S'/'M'/'B'。
        需根据你实际桶大小修改阈值
        """
        if diameter_m < 0.15:
            return 'S'
        elif diameter_m < 0.30:
            return 'M'
        else:
            return 'B'

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv5ROS2()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(str(e))
    finally:
        # 退出时关闭OpenCV窗口
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
