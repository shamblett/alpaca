/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

class GgmlInitParams {
  // Finalized to free the resource should the user not do so
  Pointer<ggmlimpl.ggml_init_params> _ptr =
      ffi.calloc<ggmlimpl.ggml_init_params>();

  set ptr(Pointer<ggmlimpl.ggml_init_params> ptr) {
    free();
    _ptr = ptr;
  }

  Pointer<ggmlimpl.ggml_init_params> get ptr => _ptr;

  ggmlimpl.ggml_init_params get instance => _ptr.ref;

  static int get size => sizeOf<ggmlimpl.ggml_init_params>();

  void free() => ffi.calloc.free(_ptr);
}
