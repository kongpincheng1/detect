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
import math
from ultralytics import YOLO
from collections import deque  # 如果使用 ultralytics 的 YOLO，可打开注释

class YOLOv5ROS2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')

        # 订阅彩色图像与深度图像
        self.color_subscription = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.depth_subscription = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)

        # 发布目标坐标及其他信息
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
        self.model.eval()
        # 或使用 ultralytics 的版本：
        # self.model = YOLO("/home/cqu/test_ws/best.pt") 

        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None

        # 相机内参（根据实际标定修改）
        self.fx = 604.7058715820312
        self.fy = 603.9238891601562
        self.cx = 321.9357604980469
        self.cy = 248.05108642578125

        # -------------------- 流程图状态机相关变量 --------------------
        # 状态机变量（状态机初始状态设为 'n3'）
        self.state = "n3"
        self.flag_a = 0  # n6：判断桶相对大小成功后置1
        self.flag_b = None
        self.flag_c = None
        self.flag_d = 0  # n16：判断是否第一次进入1桶模式
        self.flag_e = 0  # n9：判断是否第一次进入2桶模式
        self.storage1 = []  # n13：存储比较后的尺寸（第一组）
        self.storage2 = []  # n19：存储比较后的尺寸（第二组）
        self.bucket_sizes = {}
        self.DEBOUNCE_N = 4
        # 用来做桶数抖动滤波
        self.is_first_enter = True

        self.get_logger().info('YOLOv5 ROS 2 Node Initialized!')

    def image_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.depth_image is not None:
                self.process_images()
        except Exception as e:
            self.get_logger().error(f"Failed to process color image: {e}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if self.color_image is not None:
                self.process_images()
        except Exception as e:
            self.get_logger().error(f"Failed to process depth image: {e}")

    def process_images(self):
        """处理彩色图与深度图，并结合状态机流程判断桶的大小及状态转移"""

        # === 1. 获取中心深度并发布 ===
        depth_center = self.depth_image[320, 240] * 0.001
        center_height = Float32()
        center_height.data = depth_center if depth_center > 0 else 0.0
        # self.centerHeight_Pub.publish(center_height)

        # === 2. YOLOv5目标检测 ===
        detections1 = self.detect_objects(self.color_image)
        detections = self.filter_edge_detections_with_depth(detections1)
        # 假设检测结果中的 cls 对应的类别为桶（可以根据具体类别数值过滤）
        bucket_detections = [det for det in detections]
        bucket_count = len(bucket_detections)

        # === 3. 根据检测结果更新状态机 ===
        # 这里你需要将状态机中基于 input 的判断，替换为使用检测结果、尺寸比较等逻辑。
        # 例如：
        self.run_state_machine(bucket_detections, bucket_count)

        # === 4. 遍历检测目标并做可视化标注 ===
        rgb_copy = self.color_image.copy()
        edge_margin = 10
        cv2.rectangle(rgb_copy, (int(edge_margin), int(edge_margin)), (int(640-edge_margin), int(480-edge_margin)), (0,255,0), 1)

        for det, (size_label,_) in self.bucket_sizes.items():
            x1, y1, x2, y2, conf, cls = det
            # 在桶的位置画框
            cv2.rectangle(rgb_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
            # 标注桶的大小类型（B、M、S）
            cv2.putText(rgb_copy, size_label, (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Detection", rgb_copy)
        cv2.waitKey(1)

    @torch.no_grad
    def detect_objects(self, image):
        results = self.model(image, size=320)
        detections = results.xyxy[0].cpu().numpy()  # [N, 6] (x1, y1, x2, y2, conf, cls)
        detections = detections[detections[:, 4] > 0.4]  # 过滤低置信度
        return detections

    def is_bucket(self, detection):
        """
        根据检测到的类别或尺寸判断当前目标是否为桶对象。
        此处仅为示例，你需要根据模型输出或类别号进行修改。
        """
        _, _, _, _, conf, cls = detection
        # 假设 cls==0 表示桶目标，或者通过尺寸过滤
        return int(cls) == 0


    def run_state_machine(self, bucket_detections, bucket_count):
        """
        实现状态机逻辑，与原来的流程图基本一致。输入为
        - bucket_detections：检测到的桶列表（含尺寸、位置等信息）
        - bucket_count：桶的个数
        """
        # 定义需要重新识别的状态

        while True:
            if self.state == "n3":
                if bucket_count == 3:
                    self.is_first_enter = False 
                    self.get_logger().info("【n3】检测到3个桶，进入 n4")
                    self.state = "n4"
                elif bucket_count ==2 and not self.is_first_enter:
                    self.get_logger().info("【n3】检测到3个桶，执行 n1（识别）")
                    self.state = "n9"
                else:
                    self.state = "n3"
                    break
                    

            elif self.state == "n1":
                # 这里可进行图像识别补充逻辑
                self.get_logger().info("【n1】进行图像识别后返回 n3")
                self.state = "n3"
                break  # 需要重新识别，退出循环

            elif self.state == "n4":
                # n4：重置存储及flag
                self.storage1 = []
                self.storage2 = []
                self.flag_d = 0
                self.flag_e = 0
                self.get_logger().info("【n4】重置存储数据，进入 n5")
                self.state = "n5"

            elif self.state == "n5":
                # n5：根据桶尺寸判断相对大小
                bucket_diameters = [self.estimate_diameter(det) for det in bucket_detections]
                valid_diameters = [d for d in bucket_diameters if d > 0]  # 过滤无效尺寸
                if len(valid_diameters) != 3:
                    self.get_logger().info("【n5】判断失败，返回 n3")
                    self.state = "n3"
                    break  # 需要重新识别，退出循环
                else:
                    self.bucket_sizes = self.classify_buckets(bucket_detections)
                    self.flag_a = 1
                    self.get_logger().info("【n6】判断成功，flag_a置1，进入 n7")
                    self.state = "n7"
                    break

            elif self.state == "n7":
                # n7：判断画面是否为2个桶
                if bucket_count != 2:
                    self.get_logger().info(f"【n7】桶数:{bucket_count},不为2，返回 n3")
                    self.state = "n3"
                    break  # 需要重新识别，退出循环
                else:
                    self.get_logger().info("【n7】检测到2个桶，进入 n8")
                    self.state = "n8"
                
            # elif self.state == "n-1": 
            #     if bucket_count != 2:
            #         self.state="n3"
            #         break
            #     else:
            #         self.get_logger().info("【n-1】检测到2个桶，进入 n8")
            #         self.state = "n8"

            elif self.state == "n8":
                # n8：处理2桶模式，设置 flag_b（这里直接置0），进入 n9
                self.flag_b = 0
                self.get_logger().info("【n8】2桶模式，设置 flag_b=0，进入 n9")
                self.state = "n9"

            elif self.state == "n9":
                # n9：判断是否第一次进入2桶模式（即 flag_e 是否为1）
                if self.flag_e == 1:
                    self.get_logger().info("【n10】已进入2桶模式（flag_e==1），沿用初始判断")
                    self.bucket_sizes = self.adopt_storage1(bucket_detections)
                    self.state = "n14"
                    break
                else:
                    self.get_logger().info("【n9】第一次进入2桶模式，进入 n11")
                    self.state = "n11"

            elif self.state == "n11":
                # n11：比较上一帧数据确定大小
                if not self.compare_with_previous_frame_2(bucket_detections):
                    self.get_logger().info("【n11】比较失败，返回 n7")
                    self.state = "n7"
                    break
                else:
                    self.flag_e = 1
                    self.get_logger().info("【n12】比较成功，设置 flag_e=1，进入 n13")
                    self.state = "n13"

            elif self.state == "n13":
                # n13：存储比较后的尺寸
                # if bucket_detections:
                #     self.storage1 = self.estimate_diameter(bucket_detections[0])  # 示例存储第一个桶的直径
                for _, (size_label, _) in self.bucket_sizes.items():
                    self.storage1.append(size_label)
                self.get_logger().info("【n13】存储尺寸数据1，进入 n14")
                self.state = "n14"
                break

            elif self.state == "n14":
                # n14：重新检测桶的个数，决定下一步分支
                if bucket_count == 2:
                    self.get_logger().info("【n14】检测到2个桶，返回 n8")
                    self.state = "n8"
                elif bucket_count == 3:
                    self.get_logger().info("【n14】检测到3个桶，返回 n4")
                    self.state = "n4"
                elif bucket_count == 1:
                    self.get_logger().info("【n14】检测到1个桶，进入 n15")
                    self.state = "n15"
                else:
                    self.get_logger().info("【n14】桶数不为1、2、3，返回 n3")
                    self.state = "n3"
                    break  # 需要重新识别，退出循环

            elif self.state == "n15":
                # n15：1桶模式，设置 flag_c=0
                self.flag_c = 0
                self.get_logger().info("【n15】检测到1个桶，设置 flag_c=0，进入 n16")
                self.state = "n16"

            elif self.state == "n16":
                # n16：判断1桶模式是否第一次进入（flag_d）
                if self.flag_d != 1:
                    self.get_logger().info("【n16】第一次进入1桶模式，进入 n17")
                    self.state = "n17"
                else:
                    self.get_logger().info("【n16】非第一次进入1桶模式，进入 n20")
                    self.state = "n20"

            elif self.state == "n17":
                # n17：比较上一帧数据确定1桶模式下大小
                if not self.compare_with_previous_frame_1(bucket_detections):
                    self.get_logger().info("【n17】比较失败，返回 n14")
                    self.state = "n14"
                    break
                else:
                    self.flag_d = 1
                    self.get_logger().info("【n18】比较成功，设置 flag_d=1，进入 n19")
                    self.state = "n19"

            elif self.state == "n19":
                # n19：存储1桶模式比较后的尺寸
                # if bucket_detections:
                #     self.storage2 = self.estimate_diameter(bucket_detections[0])
                for _, (size_label, _) in self.bucket_sizes.items():
                    self.storage2.append(size_label)
                self.get_logger().info("【n19】存储尺寸数据至 storage2，进入 n21")
                self.state = "n21"
                break

            elif self.state == "n20":
                # n20：沿用第一次的判断值
                self.get_logger().info("【n20】已在1桶模式，沿用初始判断，进入 n21")
                self.bucket_sizes = self.adopt_storage2(bucket_detections)
                self.state = "n21"
                break

            elif self.state == "n21":
                # n21：重新检测桶的个数并分支
                if bucket_count == 1:
                    self.get_logger().info("【n21】检测到1个桶，返回 n15")
                    self.state = "n15"
                elif bucket_count == 3:
                    self.get_logger().info("【n21】检测到3个桶，返回 n4")
                    self.state = "n4"
                elif bucket_count == 2:
                    self.get_logger().info("【n21】检测到2个桶，进入 n22")
                    self.state = "n22"
                else:
                    self.get_logger().info("【n21】桶数不为1、2、3，返回 n3")
                    self.state = "n3"
                    break  # 需要重新识别，退出循环

            elif self.state == "n22":
                # n22：检测到2个桶时，重置1桶模式的相关标志，返回 n8
                self.flag_d = 0
                self.storage2 = []
                self.get_logger().info("【n22】2桶检测到，重置 flag_d 和 storage2，返回 n8")
                self.state = "n8"

            else:
                self.get_logger().info("未知状态，重置到 n3")
                self.state = "n3"
                break  # 需要重新识别，退出循环

            # 如果当前状态是需要重新识别的状态，则退出循环
            # if self.state in recognition_states:
            #     break

        self.get_logger().info(f"当前状态: {self.state}")
    
    def classify_buckets(self, detections):
        bucket_detections = [det for det in detections]
        bucket_diameters = [self.estimate_diameter(det) for det in bucket_detections]
        valid_diameters = [d for d in bucket_diameters if d > 0]  # 过滤无效尺寸


        bucket_sizes = {}
        if valid_diameters:
            max_diameter = max(valid_diameters)
            min_diameter = min(valid_diameters)
            avg_diameter = np.mean(valid_diameters)  # 可以用平均直径来做分类依据
        for det, diameter in zip(bucket_detections, bucket_diameters):
            key = tuple(det.tolist() if isinstance(det, np.ndarray) else det)
            if diameter == max_diameter:
                size_label = "B"
            elif diameter == min_diameter:
                size_label = "S"
            else:
                size_label = "M"
            
            # 存储桶的大小分类和具体的直径
            bucket_sizes[key] = (size_label, diameter)
        return bucket_sizes
    
    def adopt_storage1(self,detections):
        bucket_detections = [det for det in detections]
        bucket_diameters = [self.estimate_diameter(det) for det in bucket_detections]
        valid_diameters = [d for d in bucket_diameters if d > 0]  # 过滤无效尺寸
        bucket_size={}
        if len(valid_diameters)!=2:
            pass
        else:
            if valid_diameters[0] > valid_diameters[1]:
                if "B" in self.storage1 and "S" in self.storage1:
                    bucket_size[tuple(bucket_detections[0].tolist())]=("B", valid_diameters[0])
                    bucket_size[tuple(bucket_detections[1].tolist())]=("S", valid_diameters[1])
                elif "B" in self.storage1 and "M" in self.storage1:
                    bucket_size[tuple(bucket_detections[0].tolist())]=("B", valid_diameters[0])
                    bucket_size[tuple(bucket_detections[1].tolist())]=("M", valid_diameters[1])
                else:
                    bucket_size[tuple(bucket_detections[0].tolist())]=("M", valid_diameters[0])
                    bucket_size[tuple(bucket_detections[1].tolist())]=("S", valid_diameters[1])
            else:
                if "B" in self.storage1 and "S" in self.storage1:
                    bucket_size[tuple(bucket_detections[0].tolist())]=("S", valid_diameters[0])
                    bucket_size[tuple(bucket_detections[1].tolist())]=("B", valid_diameters[1])
                elif "B" in self.storage1 and "M" in self.storage1:
                    bucket_size[tuple(bucket_detections[0].tolist())]=("M", valid_diameters[0])
                    bucket_size[tuple(bucket_detections[1].tolist())]=("B", valid_diameters[1])
                else:
                    bucket_size[tuple(bucket_detections[0].tolist())]=("S", valid_diameters[0])
                    bucket_size[tuple(bucket_detections[1].tolist())]=("M", valid_diameters[1])
        return bucket_size
    
    def adopt_storage2(self,detections):
        bucket_detections = [det for det in detections]
        bucket_diameters = [self.estimate_diameter(det) for det in bucket_detections]
        valid_diameters = [d for d in bucket_diameters if d > 0]  # 过滤无效尺寸
        bucket_size={}
        if len(valid_diameters)!=1:
            pass
        else:
            label=self.storage2[0]
            bucket_size[tuple(bucket_detections[0].tolist())]=(label,bucket_diameters[0])
        return bucket_size
       

        

    def estimate_diameter(self, detection):
        """
        根据检测框和深度数据估计桶的直径（真实尺寸）。
        示例中采用简单算法，你可替换为更复杂的对比/计算逻辑。
        """
        x1, y1, x2, y2, conf, cls = detection
        cx_pixel = int((x1 + x2) / 2)
        cy_pixel = int((y1 + y2) / 2)
        bbox_width_px = (x2 - x1)
        median_depth = self.get_roi_median_depth(x1, y1, x2, y2)
        if median_depth <= 0:
            median_depth = self.depth_image[cy_pixel, cx_pixel] * 0.001
        if median_depth > 0:
            return (bbox_width_px / self.fx) * median_depth

        return -1

    def compare_with_previous_frame_2(self, detections):
        """
        帧间比较：找到当前帧桶的直径与上一帧直径最接近的桶，
        并将当前帧桶的大小设置为最接近直径对应的大小。
        """
        # 获取当前帧检测到的桶的直径
        bucket_detections = [det for det in detections]
        bucket_diameters = [self.estimate_diameter(det) for det in bucket_detections]
        valid_diameters = [d for d in bucket_diameters if d > 0]  # 过滤无效尺寸

        if len(valid_diameters) != 2:
            # 如果当前帧的桶数量不为 2，返回 False
            return False

        # 获取上一帧的桶的直径和分类（从 bucket_sizes 中提取）
        previous_frame_diameters = {det: diameter for det, (size_label, diameter) in self.bucket_sizes.items()}

        # 清空 bucket_sizes，准备更新
        updated_bucket_sizes = {}

        # 对于当前帧的每个桶，找到与上一帧直径最接近的桶，并更新分类
        for det, current_diameter in zip(bucket_detections, valid_diameters):
            # 找到上一帧中与当前直径最接近的直径
            closest_previous_diameter = min(previous_frame_diameters.values(), key=lambda x: abs(x - current_diameter))
            
            # 根据最接近的直径来获取分类（大小）
            for previous_det, (previous_size_label, previous_diameter) in self.bucket_sizes.items():
                key = tuple(det.tolist() if isinstance(det, np.ndarray) else det)
                if previous_diameter == closest_previous_diameter:
                    # 更新当前桶的大小分类为与最接近直径对应的分类
                    updated_bucket_sizes[key] = (previous_size_label, current_diameter)
        self.bucket_sizes.clear()
        self.bucket_sizes.update(updated_bucket_sizes)
        return True
    
    def compare_with_previous_frame_1(self, detections):
        """
        帧间比较：找到当前帧桶的直径与上一帧直径最接近的桶，
        并将当前帧桶的大小设置为最接近直径对应的大小。
        """
        # 获取当前帧检测到的桶的直径
        bucket_detections = [det for det in detections]
        bucket_diameters = [self.estimate_diameter(det) for det in bucket_detections]
        valid_diameters = [d for d in bucket_diameters if d > 0]  # 过滤无效尺寸

        if len(valid_diameters) != 1:
            # 如果当前帧的桶数量不为 2，返回 False
            return False

        # 获取上一帧的桶的直径和分类（从 bucket_sizes 中提取）
        previous_frame_diameters = {det: diameter for det, (size_label, diameter) in self.bucket_sizes.items()}

        # 清空 bucket_sizes，准备更新
        updated_bucket_sizes = {}

        # 对于当前帧的每个桶，找到与上一帧直径最接近的桶，并更新分类
        for det, current_diameter in zip(bucket_detections, valid_diameters):
            # 找到上一帧中与当前直径最接近的直径
            closest_previous_diameter = min(previous_frame_diameters.values(), key=lambda x: abs(x - current_diameter))
            
            # 根据最接近的直径来获取分类（大小）
            for previous_det, (previous_size_label, previous_diameter) in self.bucket_sizes.items():
                key = tuple(det.tolist() if isinstance(det, np.ndarray) else det)
                if previous_diameter == closest_previous_diameter:
                    # 更新当前桶的大小分类为与最接近直径对应的分类
                    updated_bucket_sizes[key] = (previous_size_label, current_diameter)
        self.bucket_sizes.clear()
        self.bucket_sizes.update(updated_bucket_sizes)
        return True

    def pixel_to_world(self, u, v, depth):
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z

    def get_roi_median_depth(self, x1, y1, x2, y2):
        h, w = self.depth_image.shape[:2]
        x1 = max(0, min(w-1, int(x1)))
        x2 = max(0, min(w-1, int(x2)))
        y1 = max(0, min(h-1, int(y1)))
        y2 = max(0, min(h-1, int(y2)))
        roi = self.depth_image[y1:y2, x1:x2].flatten()
        roi_valid = roi[(roi>0) & (roi<10000)]
        if len(roi_valid)==0:
            return -1.0
        med_mm = np.median(roi_valid)
        return med_mm*0.001

    def filter_edge_detections_with_depth(self, detections):

        filtered_detections = []
     
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            
            # 计算目标框与图像边缘的距离（单位：像素）
            distance_to_left = x1
            distance_to_right = 640-x2
            distance_to_top = y1
            distance_to_bottom = 480-y2
            
            # 判断目标是否接近图像边缘（根据深度信息调整的动态边缘距离）
            if distance_to_bottom >10 and distance_to_right >10 and distance_to_left>10 and distance_to_top>10:
                filtered_detections.append(det)

        return filtered_detections

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