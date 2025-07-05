import warnings
warnings.simplefilter('ignore', category=FutureWarning)

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOv5ROS2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')

        # --- 参数声明 ---
        self.declare_parameter('weights_path', '/home/weights/best.engine')
        self.declare_parameter('conf_threshold', 0.4)
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('cam.fx', 604.7058715820312)
        self.declare_parameter('cam.fy', 603.9238891601562)
        self.declare_parameter('cam.cx', 321.9357604980469)
        self.declare_parameter('cam.cy', 248.05108642578125)

        # --- 获取参数 ---
        weights_path = self.get_parameter('weights_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.fx = self.get_parameter('cam.fx').get_parameter_value().double_value
        self.fy = self.get_parameter('cam.fy').get_parameter_value().double_value
        self.cx = self.get_parameter('cam.cx').get_parameter_value().double_value
        self.cy = self.get_parameter('cam.cy').get_parameter_value().double_value

        # --- 发布者 ---
        self.publisher = self.create_publisher(Point, '/target_position', 10)
        self.centerHeight_Pub = self.create_publisher(Float32, '/current_height', 10)

        # --- 模型加载 ---
        self.model = YOLO(weights_path)

        # --- OpenCV 桥接 ---
        self.bridge = CvBridge()

        # --- 消息同步 ---
        color_sub = message_filters.Subscriber(self, Image, color_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        self.get_logger().info('YOLOv5 ROS 2 Node Initialized!')

    def synced_callback(self, color_msg, depth_msg):
        """
        同步后的图像回调
        """
        try:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            self.process_images(color_image, depth_image)
        except Exception as e:
            self.get_logger().error(f"Failed to process synced images: {e}")

    def process_images(self, color_image, depth_image):
        """
        同时处理彩色图像和深度图像
        """
        # --- 获取并发布中心点高度 ---
        height, width = depth_image.shape
        center_y, center_x = height // 2, width // 2
        depth_center = depth_image[center_y, center_x] * 0.001
        
        center_height = Float32()
        center_height.data = float(depth_center)
        self.centerHeight_Pub.publish(center_height)

        # --- YOLO 目标检测 ---
        results = self.detect_objects(color_image)
        
        # --- (可选) 调试显示 ---
        self.show_detections(color_image.copy(), results)

        # --- 处理每个检测结果 ---
        for result in results:
            x1, y1, x2, y2, conf, cls = result
            center_x_pixel = int((x1 + x2) / 2)
            center_y_pixel = int((y1 + y2) / 2)

            # 获取稳健的深度值
            depth = self.get_robust_depth(depth_image, center_x_pixel, center_y_pixel)
            
            if depth > 0:
                # 计算真实世界坐标
                X, Y, Z = self.pixel_to_world(center_x_pixel, center_y_pixel, depth)
                # 发布坐标
                self.process_and_publish(X, Y, Z)

    def show_detections(self, image, detections):
        """用于调试，显示检测框"""
        for det in detections:
            x1, y1, x2, y2, _, _ = det
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imshow("Detection", image)
        cv2.waitKey(1)

    @torch.no_grad()
    def detect_objects(self, image):
        results = self.model(image)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls = float(box.cls[0])
            if conf > self.conf_threshold:
                detections.append([x1, y1, x2, y2, conf, cls])
        return np.array(detections)

    def get_robust_depth(self, depth_image, x, y, size=5):
        h, w = depth_image.shape
        x1 = max(0, x - size // 2)
        x2 = min(w - 1, x + size // 2)
        y1 = max(0, y - size // 2)
        y2 = min(h - 1, y + size // 2)

        patch = depth_image[y1:y2+1, x1:x2+1]
        valid_depths = patch[patch > 0]
        
        if valid_depths.size > 0:
            return np.median(valid_depths) * 0.001
        return 0.0

    def pixel_to_world(self, u, v, depth):
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z

    def process_and_publish(self, X, Y, Z):
        # 注意: 此处的坐标变换非常依赖于你的具体应用
        # y=Y-0.05 可能是校准偏移
        # z=0.0 是将点投影到XY平面，这会丢失高度信息。
        # 如果需要3D信息，应该使用Z。
        point_msg = Point(x=X, y=Y, z=Z) # 使用计算出的 Z
        self.publisher.publish(point_msg)
        self.get_logger().info(f'Published Target: X={point_msg.x:.3f}, Y={point_msg.y:.3f}, Z={point_msg.z:.3f}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv5ROS2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.')
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
