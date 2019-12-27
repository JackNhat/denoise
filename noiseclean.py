import os
import pickle

import tensorflow as tf
import numpy as np
import scipy.special as spsp

from dev.ResNet import ResNet
from dev.acoustic.feat import polar
from dev.acoustic.analysis_synthesis.polar import synthesis
from dev import utils
import dev.optimisation as optimisation
from dev.se_batch import Train_list, Batch
import dev.gain as gain


N_w = int(16000 * 32 * 0.001)  # window length (samples).
N_s = int(16000 * 16 * 0.001)  # window shift (samples).
NFFT = int(pow(2, np.ceil(np.log2(N_w))))  # number of DFT components.

root_path = os.getcwd()

with open(f'{root_path}/data' + '/stats.p', 'rb') as f:
    stats = pickle.load(f)


class DeepXiNet:
    def __init__(self):
        print('Preparing graph...')

        # RESNET
        self.input_ph = tf.placeholder(tf.float32, shape=[None, None, 257],
                                       name='input_ph')  # noisy speech MS placeholder.
        # noisy speech MS sequence length placeholder.
        self.nframes_ph = tf.placeholder(tf.int32, shape=[None],
                                         name='nframes_ph')
        self.output = ResNet(self.input_ph, self.nframes_ph, 'FrameLayerNorm', n_blocks=40,
                             boolean_mask=True, d_out=257,
                             d_model=256, d_f=64, k_size=3,
                             max_d_rate=16)

        # TRAINING FEATURE EXTRACTION GRAPH
        self.s_ph = tf.placeholder(tf.int16, shape=[None, None],
                                   name='s_ph')  # clean speech placeholder.
        self.d_ph = tf.placeholder(tf.int16, shape=[None, None], name='d_ph')  # noise placeholder.
        self.s_len_ph = tf.placeholder(tf.int32, shape=[None],
                                       name='s_len_ph')  # clean speech sequence length placeholder.
        self.d_len_ph = tf.placeholder(tf.int32, shape=[None],
                                       name='d_len_ph')  # noise sequence length placeholder.
        self.snr_ph = tf.placeholder(tf.float32, shape=[None], name='snr_ph')  # SNR placeholder.
        self.train_feat = polar.input_target_xi(self.s_ph, self.d_ph, self.s_len_ph,
                                                self.d_len_ph, self.snr_ph, N_w, N_s,
                                                NFFT, 16000, stats['mu_hat'],
                                                stats['sigma_hat'])

        # INFERENCE FEATURE EXTRACTION GRAPH
        self.infer_feat = polar.input(self.s_ph, self.s_len_ph, N_w, N_s, NFFT,
                                      16000)

        # PLACEHOLDERS
        self.x_ph = tf.placeholder(tf.int16, shape=[None, None],
                                   name='x_ph')  # noisy speech placeholder.
        self.x_len_ph = tf.placeholder(tf.int32, shape=[None],
                                       name='x_len_ph')  # noisy speech sequence length placeholder.
        self.target_ph = tf.placeholder(tf.float32, shape=[None, 257],
                                        name='target_ph')  # training target placeholder.
        self.keep_prob_ph = tf.placeholder(tf.float32,
                                           name='keep_prob_ph')  # keep probability placeholder.
        self.training_ph = tf.placeholder(tf.bool, name='training_ph')  # training placeholder.

        # SYNTHESIS GRAPH
        self.infer_output = tf.nn.sigmoid(self.output)
        self.y_MAG_ph = tf.placeholder(tf.float32, shape=[None, None, 257],
                                       name='y_MAG_ph')
        self.x_PHA_ph = tf.placeholder(tf.float32, [None, None, 257], name='x_PHA_ph')
        self.y = synthesis(self.y_MAG_ph, self.x_PHA_ph, N_w, N_s, NFFT)

        # LOSS & OPTIMIZER
        self.loss = optimisation.loss(self.target_ph, self.output, 'mean_sigmoid_cross_entropy',
                                      axis=[1])
        self.total_loss = tf.reduce_mean(self.loss, axis=0)
        self.trainer, _ = optimisation.optimiser(self.total_loss, optimizer='adam', grad_clip=True)

        # SAVE VARIABLES
        self.saver = tf.train.Saver(max_to_keep=256)

        # NUMBER OF PARAMETERS
        params = (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


net = DeepXiNet()
config = utils.gpu_config('0')
with tf.Session(config=config) as sess:
    net.saver.restore(sess, f'{root_path}/checkpoint_dir/epoch-173')

out_path = f'{root_path}/tmp'
if not os.path.exists(out_path):
    os.makedirs(out_path)

test_x, test_x_len, test_snr, test_fnames = Batch("audio/", '*.wav', [])

for j in range(len(test_x_len)):
    input_feat = sess.run(net.infer_feat,
                          feed_dict={net.s_ph: [test_x[j][0:test_x_len[j]]],
                                     net.s_len_ph: [
                                         test_x_len[j]]})  # sample of training set.

    xi_mapped_hat = sess.run(net.infer_output, feed_dict={net.input_ph: input_feat[0],
                                                          net.nframes_ph: input_feat[1],
                                                          net.training_ph: False})
    xi_dB_hat = np.add(np.multiply(np.multiply(stats['sigma_hat'], np.sqrt(2.0)),
                                   spsp.erfinv(np.subtract(np.multiply(2.0, xi_mapped_hat), 1))),
                       stats['mu_hat'])
    xi_hat = np.power(10.0, np.divide(xi_dB_hat, 10.0))

    y_MAG = np.multiply(input_feat[0], gain.gfunc(xi_hat, xi_hat + 1, gtype='mmse-lsa'))
    y = np.squeeze(sess.run(net.y, feed_dict={net.y_MAG_ph: y_MAG,
                                              net.x_PHA_ph: input_feat[2],
                                              net.nframes_ph: input_feat[1],
                                              net.training_ph: False}))
    if np.isnan(y).any():
        ValueError('NaN values found in enhanced speech.')
    if np.isinf(y).any():
        ValueError('Inf values found in enhanced speech.')
    utils.save_wav(out_path + '/' + test_fnames[j] + '.wav', 16000, y)

