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



class YOLOv5ROS2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')

        # 订阅彩色图像
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # 彩色图像话题
            self.image_callback,
            10
        )

        # 订阅深度图像
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',  # 深度图像话题
            self.depth_callback,
            10
        )

        # 发布目标真实世界坐标
        self.publisher = self.create_publisher(Point, '/target_position', 10)
        self.centerHeight_Pub = self.create_publisher(Float32, '/current_height',10)

        #--------------------模型加载------------------------


        weights_path = Path('/home/cqu/test_ws/src/detect/best.pt')
     

        self.model = torch.hub.load(
            repo_or_dir='/home/cqu/yolov5',
            model='custom',
            path=weights_path,
            source='local',
            force_reload=True
        )


        self.model.eval()

        #--------------------------------------------------------------

        # OpenCV 桥接
        self.bridge = CvBridge()

        # 用于保存最新的深度图和彩色图
        self.color_image = None
        self.depth_image = None
        
        self.alpha = 0.2  # 平滑系数

        # 滤波状态变量
        self.last_filtered_x = None
        self.last_filtered_y = None
        self.last_filtered_z = None

        self.get_logger().info('YOLOv5 ROS 2 Node Initialized!')

    def image_callback(self, msg):
        """
        彩色图像回调
        """
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.depth_image is not None:
                self.process_images()
        except Exception as e:
            self.get_logger().error(f"Failed to process color image: {e}")

    def depth_callback(self, msg):
        """
        深度图像回调
        """
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if self.color_image is not None:
                self.process_images()
        except Exception as e:
            self.get_logger().error(f"Failed to process depth image: {e}")

    def process_images(self):
        """
        同时处理彩色图像和深度图像
        """

        depth_center = self.depth_image[320,240]*0.001
        center_height = Float32()
        if depth_center > 0:
            center_height.data = depth_center
            # self.get_logger().warn(f"current_height:{depth_center}")
        else:
            center_height.data = 0.0
        self.centerHeight_Pub.publish(center_height)

        # YOLOv5 目标检测
        results = self.detect_objects(self.color_image)

        for result in results:
            x1, y1, x2, y2, conf, cls = result
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # 获取深度值
            depth = self.depth_image[center_y, center_x] * 0.001  # 深度值以毫米为单位，转为米
            if depth > 0:
                # 计算真实世界坐标
                X, Y, Z = self.pixel_to_world(center_x, center_y, depth)

                # 发布坐标到 ROS 话题
                # if self.should_update_target(self.last_filtered_x,self.last_filtered_y,X,Y):

                self.process_and_publish(X,Y,Z)

    # def should_update_target(self, last_x, last_y, current_x, current_y, threshold=0.01):
    #     """
    #     判断目标点是否需要更新。
    #     :param last_x, last_y, last_z: 上一次目标点坐标
    #     :param current_x, current_y, current_z: 当前计算的目标点坐标
    #     :param threshold: 更新阈值，单位：米
    #     :return: 是否需要更新
    #     """
    #     if self.last_filtered_x is None and self.last_filtered_y is None:
    #         return True
    #     else:
    #         distance = math.sqrt((current_x - last_x) ** 2 + (current_y - last_y) ** 2)
    #         self.get_logger().info("计算两点距离")
    #         return distance >= threshold    

        
    # 显示检测结果
    # annotated_frame = self.render_results(self.color_image, results)
    # cv2.imshow('YOLOv5 ROS2 Detection', annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     rclpy.shutdown()

    @torch.no_grad
    def detect_objects(self, image):
        """
        使用 YOLOv5 模型检测目标
        """
        results = self.model(image, size=320)
        detections = results.xyxy[0].cpu().numpy()  # 转为 NumPy 数组
        detections = detections[detections[:, 4] > 0.4]  # 过滤低置信度结果
        return detections

    def pixel_to_world(self, u, v, depth):
        """
        像素坐标 (u, v) 和深度值转换为真实世界坐标 (X, Y, Z)
        """
        # 假设相机内参已知，替换为你的标定结果
        fx, fy = 605.7783203125, 605.474609375  # 焦距 (单位：像素)
        cx, cy = 326.34991455078125, 242.88038635253906  # 主点 (单位：像素)
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        return X, Y, Z

    # def render_results(self, image, results):
    #     """
    #     绘制 YOLOv5 检测结果
    #     """
    #     for det in results:
    #         x1, y1, x2, y2, conf, cls = det
    #         label = f'{self.model.names[int(cls)]} {conf:.2f}'
    #         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    #         cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    #     return image

    # def exponential_smoothing(self, raw_value, last_filtered_value):
    #     """
    #     对单个值应用指数平滑滤波。
    #     :param raw_value: 当前原始值
    #     :param last_filtered_value: 上一个平滑值
    #     :return: 当前平滑值
    #     """
    #     if last_filtered_value is None:
    #         return raw_value  # 初次滤波时直接返回原始值
    #     return self.alpha * raw_value + (1 - self.alpha) * last_filtered_value

    # def limit_target_step(self, last_x, last_y, target_x, target_y, max_step=0.15):
    #         """
    #         限制目标点的相邻两次发布的最大移动距离。
    #         :param last_x, last_y, last_z: 上一次发布的目标点坐标
    #         :param target_x, target_y, target_z: 当前目标点坐标
    #         :param max_step: 最大移动距离，单位：米
    #         :return: 限制后的目标点坐标
    #         """
    #         distance = math.sqrt((target_x - last_x) ** 2 + (target_y - last_y) ** 2 )
    #         if distance > max_step:
    #             scaling_factor = max_step / distance
    #             target_x = last_x + (target_x - last_x) * scaling_factor
    #             target_y = last_y + (target_y - last_y) * scaling_factor
                
    #         return target_x, target_y, 0.0


    def process_and_publish(self, X, Y, Z):
        """
        :param X: 原始 X 坐标
        :param Y: 原始 Y 坐标
        :param Z: 原始 Z 坐标
        """
        # # 对 X、Y、Z 分别应用平滑滤波
        # if self.last_filtered_x and self.last_filtered_y:
        #     X_limit, Y_limit, _ = self.limit_target_step(self.last_filtered_x, self.last_filtered_y, X, Y)
        # else:
        #     X_limit = X
        #     Y_limit = Y
        # # 对 X、Y、Z 分别应用平滑滤波
        # self.last_filtered_x = self.exponential_smoothing(X_limit, self.last_filtered_x)
        # self.last_filtered_y = self.exponential_smoothing(Y_limit, self.last_filtered_y)

        point_msg = Point(x=X, y=Y-0.05, z=0.0)
        self.publisher.publish(point_msg)
        self.get_logger().info(f'Published: X={X:.3f}, Y={Y:.3f}, Z=0.0')


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv5ROS2()

    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(str(e))
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()





 
