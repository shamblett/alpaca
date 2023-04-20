/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

class GgmlTensor {
  // Finalized to free the resource should the user not do so
  Pointer<ggmlimpl.ggml_tensor> _ptr = ffi.calloc<ggmlimpl.ggml_tensor>();

  set ptr(Pointer<ggmlimpl.ggml_tensor> ptr) {
    _ptr = ptr;
  }

  Pointer<ggmlimpl.ggml_tensor> get ptr => _ptr;

  ggmlimpl.ggml_tensor get instance => _ptr.ref;

  static int get size => sizeOf<ggmlimpl.ggml_tensor>();

  int _dataBlockCount = 0;

  void setData(Uint8List data) {
    final Pointer<Uint8> tensorData = ffi.calloc
        .allocate<Uint8>(data.length); // Allocate a pointer large enough.
    final pointerList = tensorData.asTypedList(data
        .length); // Create a list that uses our pointer and copy in the image data.
    pointerList.setAll(0, data);
    if (_dataBlockCount == 0) {
      instance.data = tensorData.cast<Void>();
    } else {
      //
    }
    _dataBlockCount++;
  }

  void free() => ffi.calloc.free(_ptr);
}
