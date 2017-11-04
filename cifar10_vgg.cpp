#include "mxnet-cpp/MxNetCpp.h"
#include <chrono>

using namespace mxnet::cpp;

int main(int argc, char **argv) {
  LG << "begin";
  const int batch_size = 512;
  const int image_size = 32;

  Symbol vgg16 = Symbol::Load("cifar10_vgg16.json");

  auto label = Symbol::Variable("label");
  Symbol net = SoftmaxOutput(vgg16, label);

  LG << "net loaded";

  std::map<std::string, NDArray> paramters;
  NDArray::Load("image-classifier-vgg16-100.params", 0, &paramters);

  std::map<std::string, NDArray> args_map;

  Context ctx = Context::gpu();

  const std::string param_prefix("vgg0_");
  for (const auto &k : paramters) {
    const std::string name(param_prefix + k.first);
    args_map[name] = k.second.Copy(ctx);
  }

  args_map["data"] = NDArray(Shape(batch_size, 3, image_size, image_size), ctx);
  args_map["label"] = NDArray(Shape(batch_size), ctx);

  NDArray::WaitAll();

  LG << "parameters loaded";

  Executor *exec = net.SimpleBind(
      ctx, args_map, std::map<std::string, NDArray>(),
      std::map<std::string, OpReqType>(), std::map<std::string, NDArray>());

  LG << "exec bound";

  auto val_iter = MXDataIter("ImageRecordIter")
                      .SetParam("path_imglist", "test.lst")
                      .SetParam("path_imgrec", "test.rec")
                      .SetParam("data_shape", Shape(3, image_size, image_size))
                      .SetParam("batch_size", batch_size)
                      .CreateDataIter();

  LG << "data iterator setup";

  int count = 0;
  Accuracy acc;
  val_iter.Reset();
  while (val_iter.Next()) {
    LG << "batch " << count++ << std::endl;
    auto data_batch = val_iter.GetDataBatch();
    data_batch.data.CopyTo(&args_map["data"]);
    data_batch.label.CopyTo(&args_map["label"]);
    NDArray::WaitAll();
    exec->Forward(false);
    acc.Update(data_batch.label, exec->outputs[0]);
  }

  LG << "Accuracy: " << acc.Get();

  delete exec;
  MXNotifyShutdown();
  return 0;
}
