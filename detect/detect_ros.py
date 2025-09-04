import warnings
warnings.simplefilter('ignore', category=FutureWarning)

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

class YOLOv5ROS2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')

        # --- 参数声明 ---
        self.declare_parameter('weights_path', '/home/weights/0728.engine')
        self.declare_parameter('conf_threshold', 0.4)
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')

        self.declare_parameter('show_image', False) 
        # <<< 修改：参数名从 record_depth_video 改为 record_rgb_video，更清晰
        self.declare_parameter('record_rgb_video', False)
        self.declare_parameter('video_output_path', '/home/depth_videos')


        # --- 获取参数 ---
        weights_path = self.get_parameter('weights_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value
        # <<< 修改：获取新参数，并使用新变量名 self.record_rgb
        self.record_rgb = self.get_parameter('record_rgb_video').get_parameter_value().bool_value
        self.video_path = self.get_parameter('video_output_path').get_parameter_value().string_value

        qos_profile_target = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        qos_profile_intrinsics = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.intrinsics_received = False # 用于标记是否已收到内参
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_profile_intrinsics  # 使用这个QoS可以确保我们能收到相机节点最后一次发布的"静态"信息
        )

        # --- 发布者 ---
        self.publisher = self.create_publisher(Point, '/target_position', qos_profile_target)
        self.centerHeight_Pub = self.create_publisher(Float32, '/current_height', 10)

        # --- 模型加载 ---
        self.model = YOLO(weights_path)

        # --- OpenCV 桥接 ---
        self.bridge = CvBridge()

        # --- 消息同步 ---
        color_sub = message_filters.Subscriber(self, Image, color_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.04
        )
        self.ts.registerCallback(self.synced_callback)

        # --- 视频录制相关初始化 ---
        self.video_writer = None
        self.is_recording = False
        if self.record_rgb: # <<< 修改：检查 self.record_rgb
            # 确保视频保存目录存在
            if not os.path.exists(self.video_path):
                os.makedirs(self.video_path)
                self.get_logger().info(f"Created video directory: {self.video_path}")

        self.get_logger().info('YOLOv5 ROS 2 Node Initialized!')
        if self.show_image: self.get_logger().info('Debug image display is ENABLED.')
        
        # <<< 修改：更新日志信息
        if self.record_rgb:
            self.get_logger().info(f'RGB video recording is ENABLED. Output path: {self.video_path}')
        else:
            self.get_logger().info('RGB video recording is DISABLED.')


    def destroy_node(self):
        """在节点销毁时调用的清理函数，确保资源被释放"""
        self.get_logger().info("Node is shutting down, attempting to clean up...")
        if self.video_writer is not None:
            self.get_logger().info("Releasing video writer...")
            self.video_writer.release()
            self.get_logger().info("Video writer released.")
        else:
            self.get_logger().warn("Video writer was not initialized, no video to save.")
        cv2.destroyAllWindows()
        super().destroy_node()

    def camera_info_callback(self, msg):
        """
        接收一次相机内参并存储，然后销毁订阅。
        """
        if not self.intrinsics_received:
            self.fx = msg.k[0]  # K[0] is fx
            self.fy = msg.k[4]  # K[4] is fy
            self.cx = msg.k[2]  # K[2] is cx
            self.cy = msg.k[5]  # K[5] is cy
            self.intrinsics_received = True
            self.get_logger().info('Camera intrinsics received successfully!')
            self.get_logger().info(f"  fx: {self.fx}, fy: {self.fy}")
            self.get_logger().info(f"  cx: {self.cx}, cy: {self.cy}")
            # 销毁订阅，因为我们只需要这个信息一次
            self.destroy_subscription(self.camera_info_sub)

    def synced_callback(self, color_msg, depth_msg):
        if not self.intrinsics_received:
            self.get_logger().warn('Waiting for camera intrinsics, skipping frame...', throttle_duration_sec=2)
            return
        
        try:
            # 彩色图使用 bgr8 格式，它与OpenCV原生格式兼容
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            self.process_images(color_image, depth_image)
        except Exception as e:
            self.get_logger().error(f"Failed to process synced images: {e}")

    def get_unique_filename(self):
        """生成一个唯一的视频文件名，避免覆盖"""
        # <<< 修改：更改视频文件名的前缀
        base_name = "rgb_video"
        ext = ".avi"
        i = 1
        while True:
            file_name = f"{base_name}_{i}{ext}"
            full_path = os.path.join(self.video_path, file_name)
            if not os.path.exists(full_path):
                return full_path
            i += 1

    def process_images(self, color_image, depth_image):
        # --- (可选) 视频录制 ---
        # <<< 修改：整个录制逻辑现在针对 color_image
        if self.record_rgb and not self.is_recording:
            try:
                filename = self.get_unique_filename()
                fps = 15 
                # 从彩色图像获取高度和宽度
                h, w, _ = color_image.shape
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (w, h), isColor=True)
                
                if not self.video_writer.isOpened():
                    self.get_logger().error(f"!!! Failed to open video writer for file: {filename}")
                    self.record_rgb = False
                else:
                    self.is_recording = True
                    self.get_logger().info(f"SUCCESS: Started recording RGB video to {filename}")
            except Exception as e:
                self.get_logger().error(f"!!! EXCEPTION while creating video writer: {e}")
                self.record_rgb = False

        # 如果正在录制，则写入彩色图像帧
        if self.is_recording and self.video_writer is not None:
            # 直接写入彩色图像，无需任何转换
            self.video_writer.write(color_image)
            
        # --- (以下检测逻辑保持不变) ---

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
        if self.show_image:
            self.show_detections(color_image.copy(), results)

        # --- 处理每个检测结果 ---
        for result in results:
            x1, y1, x2, y2, conf, cls = result
            center_x_pixel = int((x1 + x2) / 2)
            center_y_pixel = int((y1 + y2) / 2)
            depth = self.get_robust_depth(depth_image, center_x_pixel, center_y_pixel)
            
            if depth > 0:
                X, Y, Z = self.pixel_to_world(center_x_pixel, center_y_pixel, depth)
                self.process_and_publish(X, Y, Z)

    def show_detections(self, image, detections):
        for det in detections:
            x1, y1, x2, y2, _, _ = det
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imshow("Detection", image)
        cv2.waitKey(1)

    @torch.no_grad()
    def detect_objects(self, image):
        results = self.model(image,verbose=False)[0]
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
        x1 = max(0, x - size // 2); x2 = min(w - 1, x + size // 2)
        y1 = max(0, y - size // 2); y2 = min(h - 1, y + size // 2)
        patch = depth_image[y1:y2+1, x1:x2+1]; valid_depths = patch[patch > 0]
        if valid_depths.size > 0: return np.percentile(valid_depths, 95)  * 0.001
        return 0.0

    def pixel_to_world(self, u, v, depth):
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z

    def process_and_publish(self, X, Y, Z):
        point_msg = Point(x=X, y=Y, z=Z)
        self.publisher.publish(point_msg)
        self.get_logger().info(f'Published Target: X={point_msg.x:.3f}, Y={point_msg.y:.3f}, Z={point_msg.z:.3f}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv5ROS2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
