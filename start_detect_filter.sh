#！/bin/bash
cd 
cd /home/fly_ws
source install/setup.bash
ros2 run detect test & 
DETECT_PID=$!
sleep 0.5
ros2 run detect filter &
FILTER_PID=$!

wait $DETECT_PID
wait $FILTER_PID