/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

class GgmlCGraph {
  // Finalized to free the resource should the user not do so
  Pointer<ggmlimpl.ggml_cgraph> _ptr = ffi.calloc<ggmlimpl.ggml_cgraph>();

  set ptr(Pointer<ggmlimpl.ggml_cgraph> ptr) {
    free();
    _ptr = ptr;
  }

  Pointer<ggmlimpl.ggml_cgraph> get ptr => _ptr;

  ggmlimpl.ggml_cgraph get instance => _ptr.ref;

  set instance(ggmlimpl.ggml_cgraph cGraph) => _ptr.ref = cGraph;

  static int get size => sizeOf<ggmlimpl.ggml_cgraph>();

  void free() => ffi.calloc.free(_ptr);
}
