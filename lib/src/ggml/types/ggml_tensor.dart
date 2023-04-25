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

  // SJH TODO fix this for floats
  void setDataRaw(Uint8List data) {
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
    if (dPtr != nullptr) {
      return dPtr.asTypedList(32);
    }
    return Float32List(0);
  }

  GgmlTensor getSrc0() {
    final tensor = GgmlTensor();
    tensor.ptr = instance.src0;
    return tensor;
  }

  GgmlTensor getSrc1() {
    final tensor = GgmlTensor();
    tensor.ptr = instance.src1;
    return tensor;
  }

  void free() => ffi.calloc.free(ptr);

  @override
  toString() {
    final sb = StringBuffer();
    sb.writeln('Type = ${GgmlType.type(instance.type)}');
    sb.writeln('Dimensions = ${instance.n_dims}');
    sb.writeln('Op = ${GgmlOp.op(instance.op)}');
    var tmp = instance.src0 == nullptr ? 'Null' : 'Valid';
    sb.writeln('Src0 => $tmp');
    tmp = instance.src1 == nullptr ? 'Null' : 'Valid';
    sb.writeln('Src1 => $tmp');
    sb.writeln('');
    tmp = instance.data == nullptr ? 'Null' : 'Valid';
    sb.write('Data => $tmp');
    if (instance.data != nullptr) {
      final dPtr = instance.data.cast<Float>();
      tmp =
          ' (${dPtr[0].toInt()}, ${dPtr[1].toInt()}, ${dPtr[2].toInt()}, ${dPtr[3].toInt()}...)';
      sb.writeln(tmp);
    } else {
      sb.writeln('');
    }
    return sb.toString();
  }
}
