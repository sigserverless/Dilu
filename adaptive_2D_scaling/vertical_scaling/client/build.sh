#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

function build() {
    ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
    rm -rf ${ROOT}/build
    mkdir ${ROOT}/build
    cd ${ROOT}/build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

    rm -rf ${ROOT}/client-lib 
    mkdir ${ROOT}/client-lib
    cd ${ROOT}/client-lib
    cp ${ROOT}/build/libcuda-control-container-client.so ./libcuda-control.so

    touch ./ld.so.preload
    echo -e "/libcontroller.so\n/libcuda.so\n/libcuda.so.1\n/libnvidia-ml.so\n/libnvidia-ml.so.1" > ./ld.so.preload
    cp libcuda-control.so ./libnvidia-ml.so.1
    patchelf --set-soname libnvidia-ml.so.1 ./libnvidia-ml.so.1
    cp libcuda-control.so ./libnvidia-ml.so
    patchelf --set-soname libnvidia-ml.so ./libnvidia-ml.so
    cp libcuda-control.so ./libcuda.so.1
    patchelf --set-soname libcuda.so.1 ./libcuda.so.1
    cp libcuda-control.so ./libcuda.so
    patchelf --set-soname libcuda.so ./libcuda.so
    cp libcuda-control.so ./libcontroller.so
    patchelf --set-soname libcontroller.so ./libcontroller.so


    cd ..
}

build