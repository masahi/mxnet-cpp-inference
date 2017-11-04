project(mxnet-cpp-inference)

set (CMAKE_CXX_STANDARD 11)

set(MXNET_ROOT /home/masa/incubator-mxnet/)

include_directories(
  ${MXNET_ROOT}/include    
  ${MXNET_ROOT}/cpp-package/include
  ${MXNET_ROOT}/mshadow
  ${MXNET_ROOT}/dmlc-core/include
  ${MXNET_ROOT}/nnvm/include
  ${MXNET_ROOT}/dlpack/include  
  )

link_directories(${MXNET_ROOT}/lib)

set(LIBRARIES mxnet)

add_executable(cifar10_vgg cifar10_vgg.cpp)
target_link_libraries(cifar10_vgg ${LIBRARIES})
