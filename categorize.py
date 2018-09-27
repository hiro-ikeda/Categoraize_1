import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
from keras.datasets import cifar10


if __name__ == '__main__' :

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)


    nclasses = 10

    pos = 1

    for targetClass in range(nclasses) :
        targetIdx = []

        for i in range(len(y_train)) :
            if y_train[i][0] == targetClass :
                targetIdx.append(i)

        np.random.shuffle(targetIdx)

        for idx in targetIdx[:10]:
            img = toimage(X_train[idx])
            plt.subplot(10, 10, pos)
            plt.imshow(img)
            plt.axis('off')
            pos += 1

    plt.show()
