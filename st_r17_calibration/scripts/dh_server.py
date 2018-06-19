class DHServer(object):
    def __init__(self):
        self._noise = rospy.get_param('~noise', 0.0)

def main():
    rospy.init_node('dh_server')
    app = DHServer()
    app.run()

if __name__ == "__main__":
    main()
