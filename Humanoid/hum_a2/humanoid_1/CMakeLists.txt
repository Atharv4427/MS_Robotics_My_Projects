 cmake_minimum_required(VERSION 2.8.3)
 project(humanoid_1)
 
 ## Find catkin and any catkin packages
 find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg)
 
 ## Declare ROS messages and services
 ## Generate added messages and services
 generate_messages(DEPENDENCIES std_msgs)
 
 ## Declare a catkin package
 catkin_package()
 
 ## Build talker and listener
 include_directories(include ${catkin_INCLUDE_DIRS})

 catkin_install_python(PROGRAMS src/plt_imu.py
 DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(rnp_bag src/rnp_bag.cpp)
target_link_libraries(rnp_bag ${catkin_LIBRARIES})