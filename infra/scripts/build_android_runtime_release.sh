[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

# prepare ndk
if [ ! -n "$NDK_DIR" ]; then
  export NDK_DIR=$ROOT_PATH/tools/cross/ndk/r20/ndk
  echo "It will use default external path"
fi

export TARGET_OS=android
export CROSS_BUILD=1
export BUILD_TYPE=release
make -f Makefile.template install
