import numpy as np
from PIL import Image
import os
import efficient_gan
import cupy

import chainer
from chainer import functions as F
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable

root_dir = './efficient_gan_result'
out_model_dir = root_dir + '/out_models'

train, test = chainer.datasets.get_mnist(ndim=3, scale=2.0)
txs, tts = test._datasets

txs = txs - 1.0

nz = 200          # # of dim for Z
batchsize=100
n_test = len(dataset)
remain_batchsize = n_test % batchsize
anomaly_alpha = 0.9

tts = np.where(tts != 0, 1, 0)

def test_efficient_gan(gen, dis, enc):

    for i in range(0, n_test, batchsize):
        # discriminator
        # 0: from noise
        # 1: from dataset

        if (remain_batchsize != 0) and (i == (n_train - remain_batchsize)):

            x2 = txs[i:i + remain_batchsize]

        else:

            x2 = txs[i:i + batchsize]

        x2 = Variable(cuda.to_gpu(x2))
        #print "load image start ", i

        #print "load image done"

        #encoder
        z_x = enc(x2, False)

        #generator
        x_z = gen(z_x, False)

        #discriminator
        y_x = dis.call_feature(x2, z_x, False)
        y_z = dis.call_feature(x_z, z_x, False)

        x2 = F.reshape(x2, (x2.data.shape[0], -1))
        x_z = F.reshape(x_z, (x_z.data.shape[0], -1))

        #anomaly_score

        rec_loss = F.sum(abs(x2 - x_z), axis=1)
        disf_loss = F.sum(abs(y_x - y_z), axis=1)

        anomaly_score = rec_loss * anomaly_alpha + disf_loss * (1 - anomaly_alpha)

        if i == 0:

            anomaly_score_list = cuda.to_cpu(anomaly_score.data)

        else:

            anomaly_score_list = np.vstack((anomaly_score_list, cuda.to_cpu(anomaly_score.data)))


    print ('test end')



xp = cuda.cupy
cuda.get_device(0).use()

enc = efficient_gan.Encoder()
gen = efficient_gan.Generator()
dis = efficient_gan.Discriminator()

serializers.load_hdf5(out_model_dir + '/efficient_gan_model_enc_' + str(sys.argv[1]) + '.h5', enc)
serializers.load_hdf5(out_model_dir + '/efficient_gan_model_gen_' + str(sys.argv[1]) + '.h5', gen)
serializers.load_hdf5(out_model_dir + '/efficient_gan_model_dis_' + str(sys.argv[1]) + '.h5', dis)

enc.to_gpu()
gen.to_gpu()
dis.to_gpu()

test_efficient_gan(gen, dis, enc)
