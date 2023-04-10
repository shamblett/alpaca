/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

class GgmlOptParams {
  // Finalized to free the resource should the user not do so
  Pointer<ggmlimpl.ggml_opt_params> _ptr =
      ffi.calloc<ggmlimpl.ggml_opt_params>();

  set ptr(Pointer<ggmlimpl.ggml_opt_params> ptr) {
    free();
    _ptr = ptr;
  }

  Pointer<ggmlimpl.ggml_opt_params> get ptr => _ptr;

  set instance(ggmlimpl.ggml_opt_params params) => _ptr.ref = params;

  ggmlimpl.ggml_opt_params get instance => _ptr.ref;

  void free() => ffi.calloc.free(_ptr);
}
