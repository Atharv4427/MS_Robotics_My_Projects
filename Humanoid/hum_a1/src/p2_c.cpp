#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include <sstream>

ros::Publisher pub_2; 

void cb(const std_msgs::Float64::ConstPtr& msg){
    _Float64 sqr = (_Float64)msg->data;
    pub_2.publish(msg);
    sqr = sqr*sqr ;
    ROS_INFO("%f\n", sqr);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "p2");
    ros::NodeHandle n;
    ros::Subscriber sub_1 = n.subscribe("Pone", 1000, cb);
    pub_2 = n.advertise<std_msgs::Float64>("Ptwo", 1000);
    ros::spin();
    return 0;
}