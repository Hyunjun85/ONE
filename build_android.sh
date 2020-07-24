#bin/bash

TARGET_OS=android \
    CROSS_BUILD=1 \
    BUILD_TYPE=release \
    NDK_DIR=/home/one/ws/one/tools/cross/ndk/r20/ndk \
    EXT_HDF5_DIR=/home/one/ws/one/prebuilt/hdf5 \
    make install