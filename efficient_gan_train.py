import numpy as np
from PIL import Image
import os
import efficient_gan

import chainer
from chainer import functions as F
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable

root_dir = './efficient_gan_result'
out_image_dir = root_dir + '/out_images'
out_model_dir = root_dir + '/out_models'

train, test = chainer.datasets.get_mnist(ndim=3, scale=2.0)
xs, ts = train._datasets

dataset = []

for i in range(0, len(ts)):

    if ts[i] == int(sys.argv[1]):

        dataset.append(xs[i] - 1.0)

dataset = np.asarray(dataset)

nz = 200          # # of dim for Z
batchsize=100
n_epoch=100
n_train = len(dataset)
remain_batchsize = n_train % batchsize
image_save_interval = 5

def train_efficient_gan_labeled(gen, dis, enc, epoch0=1):
    o_enc = optimizers.Adam(alpha=0.00001, beta1=0.5)
    o_gen = optimizers.Adam(alpha=0.00001, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.00001, beta1=0.5)

    o_enc.setup(enc)
    o_gen.setup(gen)
    o_dis.setup(dis)


    zvis = xp.random.randn(100, nz).astype(np.float32)

    for epoch in range(epoch0,n_epoch + 1):
        perm = np.random.permutation(n_train)
        sum_l_enc = np.float32(0)
        sum_l_dis = np.float32(0)
        sum_l_gen = np.float32(0)

        for i in range(0, n_train, batchsize):
            # discriminator
            # 0: from noise
            # 1: from dataset

            if (remain_batchsize != 0) and (i == (n_train - remain_batchsize)):

                x2 = dataset[i:i + remain_batchsize]

            else:

                x2 = dataset[i:i + batchsize]

            x2 = Variable(cuda.to_gpu(x2))
            z = Variable(xp.random.randn(batchsize, nz).astype(np.float32))
            #print "load image start ", i

            #print "load image done"

            #encoder
            z_x = enc(x2, True)

            #generator
            x_z = gen(z, True)

            #discriminator
            y_x = dis(x2, z_x, True)
            y_z = dis(x_z, z, True)

            y_x = F.reshape(y_x, (y_x.data.shape[0],))
            y_z = F.reshape(y_z, (y_z.data.shape[0],))

            #loss
            L_enc = F.sigmoid_cross_entropy(y_x, Variable(xp.zeros(batchsize, dtype=np.int32)))
            L_gen = F.sigmoid_cross_entropy(y_z, Variable(xp.ones(batchsize, dtype=np.int32)))
            L_dis = F.sigmoid_cross_entropy(y_x, Variable(xp.ones(batchsize, dtype=np.int32))) + F.sigmoid_cross_entropy(y_z, Variable(xp.zeros(batchsize, dtype=np.int32)))

            #print "forward done"
            o_enc.zero_grads()
            L_enc.backward()
            o_enc.update()

            o_gen.zero_grads()
            L_gen.backward()
            o_gen.update()

            o_dis.zero_grads()
            L_dis.backward()
            o_dis.update()

            sum_l_enc += L_enc.data.get()
            sum_l_gen += L_gen.data.get()
            sum_l_dis += L_dis.data.get()

            #print "backward done"

        if ((epoch % image_save_interval) == 0) or (epoch == 1):

            z = zvis
            z = Variable(z)
            x = gen(z, False)
            x = x.data.get()
            for i_ in range(100):

                pillout = Image.fromarray(np.uint8((x[i_][0] + 1.0) / 2 * 255))

                pillout.save(out_image_dir + '/out_images_epoch_' + str(epoch) + '_' + str(i_) + '.jpg', 'JPEG')


        serializers.save_hdf5("%s/efficient_gan_model_dis_%d.h5"%(out_model_dir, epoch),dis)
        serializers.save_hdf5("%s/efficient_gan_model_enc_%d.h5"%(out_model_dir, epoch),enc)
        serializers.save_hdf5("%s/efficient_gan_model_gen_%d.h5"%(out_model_dir, epoch),gen)
        serializers.save_hdf5("%s/efficient_gan_state_dis_%d.h5"%(out_model_dir, epoch),o_dis)
        serializers.save_hdf5("%s/efficient_gan_state_enc_%d.h5"%(out_model_dir, epoch),o_enc)
        serializers.save_hdf5("%s/efficient_gan_state_gen_%d.h5"%(out_model_dir, epoch),o_gen)
        print ('epoch end', epoch, sum_l_enc/n_train, sum_l_gen/n_train, sum_l_dis/n_train)



xp = cuda.cupy
cuda.get_device(0).use()

enc = efficient_gan.Encoder()
gen = efficient_gan.Generator()
dis = efficient_gan.Discriminator()

enc.to_gpu()
gen.to_gpu()
dis.to_gpu()


try:
    os.mkdir(root_dir)
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
except:
    pass

train_efficient_gan_labeled(gen, dis, enc)
