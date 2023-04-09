/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

class GgmlInitParams {
  // Finalized to free the resource should the user not do so
  static final _ptr = ffi.calloc<ggmlimpl.ggml_init_params>();

  ggmlimpl.ggml_init_params get instance {
    _finalizer.attach(free, _ptr);
    return _ptr.ref;
  }

  static final Finalizer _finalizer = Finalizer((_) => ffi.calloc.free(_ptr));

  void free() => ffi.calloc.free(_ptr);
}
