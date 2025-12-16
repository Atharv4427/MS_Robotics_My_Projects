#!/usr/bin/env python

from std_msgs.msg import Float64
import rospy
import matplotlib.pyplot as plt
x = []
y = []

def cbx(inx) :
    x.append([inx.data])
def cby(iny) :
    y.append([iny.data])

def receive() :
    rospy.init_node("plt_imu", anonymous=True)
    rospy.Subscriber("pltx", Float64, cbx)
    rospy.Subscriber("plty", Float64, cby)
    rospy.spin()

if __name__ == '__main__' :
    receive()
    plt.plot(x,y)
    plt.show()