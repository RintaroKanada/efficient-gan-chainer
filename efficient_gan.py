import chainer
from chainer import links as L
from chainer import functions as F

class Encoder(chainer.Chain):
    def __init__(self):
        super(Encoder, self).__init__(
        c0 = L.Convolution2d(1, 32, 3, stride=1, pad=1, wscale=0.02),
        c1 = L.Convolution2d(32, 64, 3, stride=2, pad=1, wscale=0.02),
        c2 = L.Convolution2d(64, 128, 3, stride=2, pad=1, wscale=0.02),
        fc3 = L.Linear(7*7*128, 200),
        bn1 = L.BatchNormalization(64),
        bn2 = L.BatchNormalization(128),
        )

        self.train = True

    def __call__(self, x, train):

        self.train = train

        if self.train:

            test = False

        else:

            test = True

        h = self.c0(x)
        h = F.leaky_relu(self.bn1(self.c1(h), test=test), slope=0.1)
        h = F.leaky_relu(self.bn2(self.c2(h), test=test), slope=0.1)
        l = self.fc3(h)

        return l

class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
        fc0 = L.Linear(200, 1024, wscale=0.02),
        fc1 = L.Linear(1024, 7*7*128, wscale=0.02),
        dc2 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02),
        dc3 = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02),
        bn0 = L.BatchNormalization(1024),
        bn1 = L.BatchNormalization(7*7*128),
        bn2 = L.BatchNormalization(64),
        )

        self.train = True

    def __call__(self, x, train):

        self.train = train

        if self.train:

            test = False

        else:

            test = True

        h = F.relu(self.bn0(self.fc0(x), test=test))
        h = F.relu(self.bn1(self.fc1(h), test=test))
        h = F.reshape(h, (h.data.shape[]))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        l = F.tanh(self.dc3(h))

        return l

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
        x_c0 = L.Convolution2d(1, 64, 4, stride=2, pad=1, wscale=0.02),
        x_c1 = L.Convolution2d(64, 64, 4, stride=2, pad=1, wscale=0.02),
        bnx_c1 = L.BatchNormalization(64),

        z_fc0 = L.Linear(200, 512),

        xz_fc0 = L.Linear(512 + 7 * 7 * 64, 1024),
        xz_fc1 = L.Linear(1024, 1),
        )

        self.train = True

    def __call__(self, x, z, train):

        self.train = train

        if self.train:

            test = False

        else:

            test = True

        h_x = F.leaky_relu(self.x_c0(x), slope=0.1)
        h_x = F.leaky_relu(self.bnx_c1(self.x_c1(h_x), test=test), slope=0.1)
        h_x = F.reshape(h_x, (h_x.data.shape[0], -1))

        h_z = F.leaky_relu(self.z_fc0(z), slope=0.1)

        h = F.concat((h_x, h_z))
        h = F.leaky_relu(self.xz_fc0(h), slope=0.1)
        l = self.xz_fc1(h)

        return l

    def call_feature(self, x, z, train):

        self.train = train

        if self.train:

            test = False

        else:

            test = True

        h_x = F.leaky_relu(self.x_c0(x), slope=0.1)
        h_x = F.leaky_relu(self.bnx_c1(self.x_c1(h_x), test=test), slope=0.1)
        h_x = F.reshape(h_x, (h_x.data.shape[0], -1))

        h_z = F.leaky_relu(self.z_fc0(z), slope=0.1)

        h = F.concat((h_x, h_z))
        h = F.leaky_relu(self.xz_fc0(h), slope=0.1)

        return h
