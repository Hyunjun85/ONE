#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage : ./push_android.sh [TARGET_FOLDER]"
  exit 1
fi

BUILD_FOLDER=./Product/aarch64-android.release/out
TARGET_FOLDER=/data/local/tmp/$1

find $BUILD_FOLDER/lib -name "libboost*.so" -delete

adb shell mkdir -p $TARGET_FOLDER/lib
echo "export LD_LIBRARY_PATH=$TARGET_FOLDER/lib" > env_android.sh
adb push env_android.sh $TARGET_FOLDER/env.sh
rm env_android.sh
adb push $BUILD_FOLDER/bin/* $TARGET_FOLDER
adb push $BUILD_FOLDER/lib/*.so $TARGET_FOLDER/lib
adb push $BUILD_FOLDER/lib/*.a $TARGET_FOLDER/lib
adb push ./tools/cross/ndk/r20/ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so $TARGET_FOLDER/lib