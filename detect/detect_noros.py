import warnings
warnings.simplefilter('ignore', category=FutureWarning)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import torch
import cv2
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path


class YOLOv5RealSense(Node):
    def __init__(self):
        super().__init__('yolov5_realsense')

        # 初始化 ROS 2 发布器
        self.publisher = self.create_publisher(Point, '/detected_coordinates', 10)

        # 初始化 RealSense 相机
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # 对齐深度到彩色流
        self.align = rs.align(rs.stream.color)
        try:
            # 启动相机
            self.pipeline.start(self.config)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to initialize RealSense pipeline: {e}")
        # self.pipeline.start(self.config)

        # 获取相机内参
        profile = self.pipeline.get_active_profile()
        color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx = color_intrinsics.fx
        self.fy = color_intrinsics.fy
        self.cx = color_intrinsics.ppx
        self.cy = color_intrinsics.ppy

        # 加载 YOLOv5 模型
        weights_path = Path('/home/ws_control/best.pt')

        self.model = torch.hub.load(
            repo_or_dir='/home/yolov5',
            model='custom',
            path=weights_path,
            source='local'
        )
        self.model.eval()
        self.bridge = CvBridge()

        self.get_logger().info('YOLOv5 RealSense Node Initialized!')

    def run(self):
        try:
            while rclpy.ok():
                # 获取对齐后的帧
                self.get_logger().info(f'Frames receive start')
                frames = self.pipeline.wait_for_frames()
                self.get_logger().info(f'Frames received: {frames}')

                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # 转换为 NumPy 数组
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # YOLOv5 检测
                results = self.detect_objects(color_image)

                for result in results:
                    x1, y1, x2, y2, conf, cls = result
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # 获取深度值
                    depth = depth_image[center_y, center_x] * 0.001  # 深度值以毫米为单位，转为米
                    if depth > 0:
                        X, Y, Z = self.pixel_to_world(center_x, center_y, depth)

                        # 发布坐标到 ROS 话题
                        point_msg = Point(x=X, y=Y, z=Z)
                        self.publisher.publish(point_msg)
                        self.get_logger().info(f'Published: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}')

                # 显示检测结果
                annotated_frame = self.render_results(color_image, results)
                cv2.imshow('YOLOv5 RealSense Detection', annotated_frame)

                # 按下 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def detect_objects(self, image):
        """
        使用 YOLOv5 模型检测目标
        """
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()  # 转为 NumPy 数组
        return detections

    def pixel_to_world(self, u, v, depth):
        """
        像素坐标 (u, v) 和深度值转换为真实世界坐标 (X, Y, Z)
        """
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z

    def render_results(self, image, results):
        """
        绘制 YOLOv5 检测结果
        """
        for det in results:
            x1, y1, x2, y2, conf, cls = det
            label = f'{self.model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return image


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv5RealSense()

    try:
        node.run()
    except Exception as e:
        node.get_logger().error(str(e))
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()