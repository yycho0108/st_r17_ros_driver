import sympy

""" DH Calibrator """
class DHCalibrator(object):
    def __init__(self, dh0, m):
        self._dh0 = np.float32(dh0)
        self._m = m
        self._initialized = False
        self._build()
        self._log()

    @staticmethod
    def _dh(index):
        """ Seed DH Parameters; note q0 is offset joint value. """
        alpha, a, d, dq, q = sympy.symbols('alpha{0}, a{0}, d{0}, dq{0}, q{0}'.format(index))
        return (alpha, a, d, q+dq), q

    @staticmethod
    def _dh2T(alpha, a, d, q, dq):
        """ Convert DH Parameters to Transformation Matrix """
        cq = cos(q + dq)
        sq = sin(q + dq)
        ca = cos(alpha)
        sa = sin(alpha)

        T = Matrix([
            [cq, -sq, 0, a],
            [sq*ca, cq*ca, -sa, -sa*d],
            [sq*sa, cq*sa, ca, ca*d],
            [0, 0, 0, 1]
            ])
        return T

    def _build(self):
        tf.reset_default_graph()
        self._graph = tf.get_default_graph()
        # inputs ...
        with tf.name_scope('inputs'):
            dhs, qs = zip(*[self._dh(i, *_dh) for (i, _dh) in enumerate(self._dh0)])
            T_f = tf.placeholder(dtype=tf.float32, shape=[None, self._m, 4,4], name='T_f') # camera_link -> object(s)
            vis = tf.placeholder(dtype=tf.bool, shape=[None, self._m], name='vis') # marker visibility
            T_targ = tf.placeholder(dtype=tf.float32, shape=[self._m, 4, 4], name='T_targ')

        # build  transformation ...
        with tf.name_scope('transforms'):
            Ts = [dh2T(*dh) for dh in dhs] # == (N, 4, 4)
            T = reduce(lambda a,b : tf.matmul(a,b), Ts) # base_link -> stereo_optical_link

            T = tf.einsum('aij,abjk->abik', T, T_f) # apply landmarks transforms
            #_T_f = tf.unstack(T_f, axis=1)
            #_Ts = [tf.matmul(T, _T) for _T in _T_f]
            #T = tf.stack(_Ts, axis=1)

        #mode = 'xyzrpy'
        mode = 'T'

        pred_xyz, pred_rpy = T2xyzrpy(T) # == (N, M, 3)
        vis_f = tf.cast(vis, tf.float32)

        #vis_sel = tf.tile(vis[..., tf.newaxis], [1,1,3])
        num_vis = tf.reduce_sum(vis_f, axis=0) # (N,M) -> (M)


        gamma = 0.9
        # WARNING :: due to how the running mean was implemented,
        # the first batch MUST contain all markers.
        # MAKE SURE THIS HAPPENS.

        if mode == 'xyzrpy':
            vis_sel = tf.greater(num_vis[..., tf.newaxis], 0) # -> (M, 1)
            vis_sel = tf.tile(vis_sel, (1,3)) # -> (M,3)

            with tf.name_scope('target_avg'):
                xyz_avg = tf.Variable(initial_value=np.zeros(shape=(self._m,3)), trainable=False, dtype=np.float32)
                rpy_avg = tf.Variable(initial_value=np.zeros(shape=(self._m,3)), trainable=False, dtype=np.float32)

                xyz_avg_1 = pred_xyz * vis_f[..., tf.newaxis]
                xyz_avg_1 = tf.reduce_sum(xyz_avg_1, axis=0) / num_vis[:, tf.newaxis]
                xyz_avg_1 = tf.where(vis_sel, xyz_avg_1, xyz_avg)

                rpy_avg_1 = pred_rpy * vis_f[..., tf.newaxis]
                rpy_avg_1 = tf.reduce_sum(rpy_avg_1, axis=0) / num_vis[:, tf.newaxis]
                rpy_avg_1 = tf.where(vis_sel, rpy_avg_1, rpy_avg)

                xyz0 = tf.assign(xyz_avg, xyz_avg_1)
                rpy0 = tf.assign(rpy_avg, rpy_avg_1)
                self._T_init = tf.group(xyz0, rpy0)
                
                new_xyz_avg = gamma * xyz_avg + (1.0 - gamma) * xyz_avg_1
                new_rpy_avg = gamma * rpy_avg + (1.0 - gamma) * rpy_avg_1

                xyz_avg_u = tf.assign(xyz_avg, new_xyz_avg)
                rpy_avg_u = tf.assign(rpy_avg, new_xyz_avg)
                T_update = [xyz_avg_u, rpy_avg_u]
        else:
            vis_sel = tf.greater(num_vis[..., tf.newaxis, tf.newaxis], 0) # -> (M, 1)
            vis_sel = tf.tile(vis_sel, (1,4,4)) # -> (M,3)

            with tf.name_scope('target_avg'):
                T_avg = tf.Variable(initial_value = np.zeros(shape=(self._m,4,4)), trainable=False, dtype=np.float32)

                T_avg_1 = T * vis_f[..., tf.newaxis, tf.newaxis]
                T_avg_1 = tf.reduce_sum(T_avg_1, axis=0) / num_vis[:, tf.newaxis]
                T_avg_1 = tf.where(vis_sel, T_avg_1, T_avg)
                #T_avg_1 = tf.reduce_mean(T, axis=0)

                T_update = [tf.assign(T_avg, gamma*T_avg + (1.0-gamma) * T_avg_1)]
                self._T_init = tf.assign(T_avg, T_avg_1) # don't forget to initialize!
        self._T_update = T_update

        # tf.assert( ... )

        with tf.control_dependencies(T_update):
            if mode == 'xyzrpy':
                loss_xyz = tf.square(pred_xyz - xyz_avg) #(N,M,3)
                loss_rpy = tf.square(pred_rpy - rpy_avg)
                loss = loss_xyz + loss_rpy
                loss = tf.reduce_sum(loss * vis_f[..., tf.newaxis]) / (3.0 * tf.reduce_sum(vis_f))
            else:
                # TODO : relative or running avg?
                #loss = tf.square(T - tf.reduce_mean(T, axis=0, keep_dims=True))
                #loss = tf.square(T - tf.expand_dims(T_avg, 0))
                loss = tf.square(T - tf.expand_dims(T_targ, 0))
                loss = tf.reduce_sum(loss * vis_f[..., tf.newaxis, tf.newaxis]) / (16.0 * tf.reduce_sum(vis_f))
            #loss = tf.reduce_mean(loss)

        # without running avg ... 
        #loss = tf.square(T - tf.reduce_mean(T, axis=0, keep_dims=True))
        #loss = tf.reduce_mean(loss)

        # build train ...
        #train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)
        train = tf.train.AdamOptimizer(learning_rate=5e-2).minimize(loss)

        # save ...
        self._T = T
        self._T_f = T_f
        self._T_targ = T_targ
        self._dhs = dhs
        self._vis = vis
        self._qs = qs
        self._loss = loss
        if mode == 'xyzrpy':
            self._loss_xyz = tf.reduce_mean(loss_xyz)
            self._loss_rpy = tf.reduce_mean(loss_rpy)
        self._train = train

    def _log(self):
        # logs ...
        #tf.summary.scalar('loss_xyz', self._loss_xyz)
        #tf.summary.scalar('loss_rpy', self._loss_rpy)
        tf.summary.scalar('loss', self._loss)

        self._writer = tf.summary.FileWriter('/tmp/dh/13', self._graph)
        self._summary = tf.summary.merge_all()

    def eval_1(self, js, xs, vis):
        feed_dict = {q:[j] for q,j in zip(self._qs, js)}
        feed_dict[self._T_f] = np.expand_dims(xs, 0) # [1, M, 4, 4] 
        feed_dict[self._vis] = np.expand_dims(vis, 0)
        return self.run(self._T, feed_dict = feed_dict)[0]

    def start(self):
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._iter = 0

    def step(self, js, xs, vis, ys):
        feed_dict = {q:j for q,j in zip(self._qs, np.transpose(js))}
        feed_dict[self._T_f] = xs
        feed_dict[self._vis] = vis
        feed_dict[self._T_targ] = ys

        if self._initialized:
            _, loss, dhs, summary = self.run(
                    [self._train, self._loss, self._dhs, self._summary],
                    feed_dict = feed_dict)
            self._writer.add_summary(summary, self._iter)
            self._iter += 1
            return loss, dhs
        else:
            _, dhs = self.run([self._T_init, self._dhs], feed_dict=feed_dict)
            self._initialized = True
            return -1, dhs

    def run(self, *args, **kwargs):
        return self._sess.run(*args, **kwargs)

