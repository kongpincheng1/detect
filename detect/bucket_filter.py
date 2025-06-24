#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from test_interface.msg import Buckets

class BucketFilter(Node):
    def __init__(self):
        super().__init__('bucket_filter')
        
        # 订阅所有桶位置
        self.subscription = self.create_subscription(
            Buckets,
            '/bucket_topic',
            self.bucket_callback,
            10
        )
        
        # 发布目标桶位置
        self.publisher = self.create_publisher(
            Point,
            '/target_position',
            10
        )
        
        self.get_logger().info('BucketFilter 节点已启动')
        
    def bucket_callback(self, msg):
        # 如果没有检测到桶，直接返回
        if not msg.buckets:
            self.get_logger().warn('未检测到任何桶')
            return
            
        # 将桶按类型分类
        small_buckets = [bucket for bucket in msg.buckets if bucket.z == 0.0]
        medium_buckets = [bucket for bucket in msg.buckets if bucket.z == 1.0]
        big_buckets = [bucket for bucket in msg.buckets if bucket.z == 2.0]
        
        self.get_logger().info(f'检测到: {len(small_buckets)}个小桶, {len(medium_buckets)}个中桶, {len(big_buckets)}个大桶')
        
        # 按优先级选择目标桶：小桶 > 中桶 > 大桶
        if small_buckets:
            target = small_buckets[0]
            self.get_logger().info(f'选择小桶(S)作为目标：位置(x={target.x:.2f}, y={target.y:.2f})')
        elif medium_buckets:
            target = medium_buckets[0]
            self.get_logger().info(f'选择中桶(M)作为目标：位置(x={target.x:.2f}, y={target.y:.2f})')
        elif big_buckets:
            target = big_buckets[0]
            self.get_logger().info(f'选择大桶(B)作为目标：位置(x={target.x:.2f}, y={target.y:.2f})')
        
        else:
            self.get_logger().warn('未检测到任何桶')
            return
        
        # 发布选定的目标桶位置
        self.publisher.publish(target)

def main(args=None):
    rclpy.init(args=args)
    filter_node = BucketFilter()
    
    try:
        rclpy.spin(filter_node)
    except KeyboardInterrupt:
        pass
    finally:
        filter_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()