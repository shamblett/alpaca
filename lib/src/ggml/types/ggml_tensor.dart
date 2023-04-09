/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

class GgmlTensor {
  // Finalized to free the resource should the user not do so
  final _ptr = ffi.calloc<ggmlimpl.ggml_tensor>();

  Pointer<ggmlimpl.ggml_tensor> get ptr => _ptr;

  ggmlimpl.ggml_tensor get instance => _ptr.ref;

  void free() => ffi.calloc.free(_ptr);
}
