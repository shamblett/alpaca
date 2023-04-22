/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

class GgmlTensor {
  // Finalized to free the resource should the user not do so
  Pointer<ggmlimpl.ggml_tensor> ptr = ffi.calloc<ggmlimpl.ggml_tensor>();

  ggmlimpl.ggml_tensor get instance => ptr.ref;

  static int get size => sizeOf<ggmlimpl.ggml_tensor>();

  void setData(Uint8List data) {
    final Pointer<Uint8> tensorData = ffi.calloc
        .allocate<Uint8>(data.length); // Allocate a pointer large enough.
    final pointerList = tensorData.asTypedList(data
        .length); // Create a list that uses our pointer and copy in the image data.
    pointerList.setAll(0, data);
    instance.data = tensorData.cast<Void>();
  }

  /// Get the first 2 floats from the tensor data
  Float32List getData() {
    final dPtr = instance.data.cast<Float>();
    return dPtr.asTypedList(32);
  }

  GgmlTensor getSrc0() {
    final tensor = GgmlTensor();
    tensor.ptr = tensor.instance.src0;
    return tensor;
  }

  GgmlTensor getSrc1() {
    final tensor = GgmlTensor();
    tensor.ptr = tensor.instance.src1;
    return tensor;
  }

  void free() => ffi.calloc.free(ptr);

  @override
  toString() =>
      'Type = ${GgmlType.type(instance.type)} nDims = ${instance.n_dims} '
      'nElements = ${instance.ne[0] * instance.ne[1] * instance.ne[2] * instance.ne[3]}';
}
