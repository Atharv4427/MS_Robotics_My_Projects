#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include <sstream>

int main(int argc, char** argv) {
    ros::init(argc, argv, "p4");
    ros::NodeHandle n;
    ros::Publisher pub_4 = n.advertise<std_msgs::Float64>("Pfour", 1000);

    while(ros::ok()) {
        std_msgs::Float64 msg;
        std::cout << "Enter Your Number  ";
        std::cin >> msg.data;
        pub_4.publish(msg);
        ros::spinOnce;
    }
    return 0;
}