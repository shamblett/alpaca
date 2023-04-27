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

  /// Get the raw data pointer
  Pointer<Void> getData() => Ggml().getData(this).cast<Void>();

  /// Get data as a Float *
  Pointer<Float> getDataF32() => Ggml().getDataF32(this);

  /// Set the data pointer
  void setData(Pointer<Void> data) => instance.data = data;

  /// Set the data pointer from a float pointer
  void setDataF32(Pointer<Float> data) => instance.data = data.cast<Void>();

  /// Set the data from another tensor
  void setDataTensor(GgmlTensor tensor) => setData(tensor.getData());

  /// Set data from a list of ints as ints
  void setDataInt(List<int> values) {
    if (instance.data == nullptr) {
      instance.data = ffi.calloc.allocate<Int32>(values.length).cast<Void>();
    }
    final dataList = Int32List.fromList(values);
    final buf = dataList.buffer.asInt32List();
    final byteData = ByteData.sublistView(buf);
    for (int i = 0; i < buf.length; i++) {
      final val = byteData.getInt32(i, Endian.big);
      Ggml().setI321d(this, i, val);
    }
  }

  /// Set data from a list of doubles as doubles
  void setDataDouble(List<double> values) {
    final valPtr = ffi.calloc.allocate<Float>(values.length); // All
    final pointerList = valPtr.asTypedList(values.length); // Create a list
    pointerList.setAll(0, values);
    instance.data = valPtr.cast<Void>();
  }

  /// Set data from a list of floats
  void setDataFloat(List<Float> values) {
    var listDouble = values.map((i) => i as double).toList();
    final valPtr = ffi.calloc.allocate<Float>(values.length); // All
    final pointerList = valPtr.asTypedList(values.length); // Create a list
    pointerList.setAll(0, listDouble);
    instance.data = valPtr.cast<Void>();
  }

  /// Set data from a list of bytes
  void setDataBytes(Uint8List data) {
    final Pointer<Uint8> tensorData = ffi.calloc
        .allocate<Uint8>(data.length); // Allocate a pointer large enough.
    final pointerList = tensorData.asTypedList(data
        .length); // Create a list that uses our pointer and copy in the image data.
    pointerList.setAll(0, data);
    instance.data = tensorData.cast<Void>();
  }

  /// Gets the top x data values as ints, returns an empty list if the data pointer is null.
  /// Defaults to the top 5. The caller must ensure the list is large enough for the
  /// value of the top parameter.
  List<int> getTopXDataInt([top = 5]) {
    final ret = <int>[];
    if (instance.data != nullptr) {
      final dPtr = getDataF32();
      for (int i = 0; i < top; i++) {
        ret.add(dPtr.elementAt(i).value.toInt());
      }
    }
    return ret;
  }

  /// Gets the top x data values as doubles, returns an empty list if the data pointer is null.
  /// Defaults to the top 5. The caller must ensure the list is large enough for the
  /// value of the top parameter.
  List<double> getTopXDataDouble([top = 5]) {
    final ret = <double>[];
    if (instance.data != nullptr) {
      for (int i = 0; i < top; i++) {
        final dPtr = instance.data.cast<Float>();
        ret.add(dPtr.elementAt(i).value);
      }
      for (int i = 0; i < top; i++) {}
    }
    return ret;
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

  String dump() {
    final sb = StringBuffer();
    sb.writeln('Type = ${GgmlType.type(instance.type)}');
    sb.writeln('Dimensions = ${instance.n_dims}');
    sb.writeln(
        'Nb => ${instance.nb[0]}, ${instance.nb[1]}, ${instance.nb[2]}, ${instance.nb[3]}');
    sb.writeln(
        'Ne => ${instance.ne[0]}, ${instance.ne[1]}, ${instance.ne[2]}, ${instance.ne[3]}');
    sb.writeln('Op = ${GgmlOp.op(instance.op)}');
    sb.writeln('Is param = ${instance.is_param == 0 ? 'False' : 'True'}');
    var tmp = instance.src0 == nullptr ? 'Null' : 'Valid';
    sb.writeln('Src0 => $tmp');
    tmp = instance.src1 == nullptr ? 'Null' : 'Valid';
    sb.writeln('Src1 => $tmp');
    tmp = instance.grad == nullptr ? 'Null' : 'Valid';
    sb.writeln('Grad => $tmp');
    sb.writeln('');
    return sb.toString();
  }

  /// Primarily for the debugger, use the dump method above for more data.
  @override
  toString() {
    final sb = StringBuffer();
    sb.write('Type = ${GgmlType.type(instance.type)}');
    sb.write(', Dimensions = ${instance.n_dims}');
    sb.write(', Op = ${GgmlOp.op(instance.op)}');
    return sb.toString();
  }
}
