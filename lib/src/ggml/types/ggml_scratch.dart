/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

class GgmlScratch {
  // Finalized to free the resource should the user not do so
  Pointer<ggmlimpl.ggml_scratch> _ptr = ffi.calloc<ggmlimpl.ggml_scratch>();

  set ptr(Pointer<ggmlimpl.ggml_scratch> ptr) {
    free();
    _ptr = ptr;
  }

  Pointer<ggmlimpl.ggml_scratch> get ptr => _ptr;

  ggmlimpl.ggml_scratch get instance => _ptr.ref;

  void free() => ffi.calloc.free(_ptr);
}
