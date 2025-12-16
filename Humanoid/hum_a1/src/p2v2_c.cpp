#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include <sstream>

ros::Publisher pub_2;
_Float64 a, b;
bool a_up = false, b_up = false;

void fuse() {
    if (a_up && b_up) {
        std_msgs::Float64 sum;
        sum.data = a + b;
        a_up = b_up = false;
        pub_2.publish(sum);
        // //std::cout << sum.data;
        ROS_INFO("%f \n", sum.data);
    }
}
void cb1(const std_msgs::Float64ConstPtr& msg) {
    a = msg->data;
    a_up = true;
    fuse();
}
void cb2(const std_msgs::Float64ConstPtr& msg) {
    b = msg->data;
    b_up = true;
    fuse();
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "p2v2");
    ros::NodeHandle n;
    pub_2 = n.advertise<std_msgs::Float64>("Ptwo", 1000);
    ros::Subscriber sub_1 = n.subscribe("Pone", 1000, cb1);
    ros::Subscriber sub_4 = n.subscribe("Pfour", 1000, cb2);
    ros::spin();
    return 0;
}