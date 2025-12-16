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

add_executable(p1_c src/p1_c.cpp)
target_link_libraries(p1_c ${catkin_LIBRARIES})

add_executable(p2_c src/p2_c.cpp)
target_link_libraries(p2_c ${catkin_LIBRARIES})

add_executable(p3_c src/p3_c.cpp)
target_link_libraries(p3_c ${catkin_LIBRARIES})

add_executable(p4_c src/p4_c.cpp)
target_link_libraries(p4_c ${catkin_LIBRARIES})

add_executable(p2v2_c src/p2v2_c.cpp)
target_link_libraries(p2v2_c ${catkin_LIBRARIES})