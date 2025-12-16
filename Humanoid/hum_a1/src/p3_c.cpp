#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include<sstream>

void cb(const std_msgs::Float64::ConstPtr& msg) {
    // std::cout << msg->data ;
    ROS_INFO("%f \n", msg->data);
}

int main (int argc, char** argv){
    ros::init(argc, argv, "p3");
    ros::NodeHandle n;
    ros::Subscriber sub_2 = n.subscribe("Ptwo", 1000, cb);
    ros::spin();
}