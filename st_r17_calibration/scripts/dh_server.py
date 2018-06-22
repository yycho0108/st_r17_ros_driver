class DHServer(object):
    def __init__(self):
        noise = rospy.get_param('~noise', 0.0)
        dh0   = rospy.get_param('~dh0', default=None)
        if self._dh is None:
            raise ValueError('Improper DH Parameters Input : {}'.format(self._dh))
        rospy

def main():
    rospy.init_node('dh_server')
    app = DHServer()
    app.run()

if __name__ == "__main__":
    main()
