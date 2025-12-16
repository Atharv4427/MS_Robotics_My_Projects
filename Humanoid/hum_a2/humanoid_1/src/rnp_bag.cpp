#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include "sensor_msgs/Imu.h"
#include <math.h>
#include <sstream>

ros::Publisher pub_plt_x, pub_plt_y;
float dt = 0.1;
_Float64 theta = 0.0, omega0=0.0;
_Float64 x=0.0, y=0.0, vx=0.0, vy=0.0, ax0=0.0, ay0=0.0, time_pre=0.0, vx0=0.0, vy0=0.0;

_Float64 integrate(_Float64 a, _Float64 b, _Float64 c) {
    return (a + b)*dt*0.5 + c;
}

void cb(const sensor_msgs::Imu::ConstPtr& msg){
    theta = integrate(msg->angular_velocity.x, omega0, theta);
    omega0 = msg->angular_velocity.x;
    _Float64 ax = msg->linear_acceleration.z*cos(theta) - msg->linear_acceleration.y*sin(theta);
    _Float64 ay = msg->linear_acceleration.z*sin(theta) + msg->linear_acceleration.y*cos(theta);
    vx = integrate(ax, ax0, vx);
    vy = integrate(ay, ay0, vy);
    ax0  = ax;
    ay0 = ay;
    x = integrate(vx, vx0, x);
    y = integrate(vy, vy0, y);
    vx0 = vx;
    vy0 = vy;
    printf("[%f, %f], \n", x, y);
    std_msgs::Float64 px, py;
    px.data = x;
    py.data = y;
    pub_plt_x.publish(px);
    pub_plt_y.publish(py);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "rnp");
    ros::NodeHandle n;
    pub_plt_x = n.advertise<std_msgs::Float64>("pltx", 1000);
    pub_plt_y = n.advertise<std_msgs::Float64>("plty", 1000);
    ros::Subscriber sub_bag = n.subscribe("imu", 1000, cb);
    ros::spin();
    return 0;
}