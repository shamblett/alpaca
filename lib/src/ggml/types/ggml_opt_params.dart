/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

class GgmlOptParams {
  // Finalized to free the resource should the user not do so
  Pointer<ggmlimpl.ggml_opt_params> ptr =
      ffi.calloc<ggmlimpl.ggml_opt_params>();

  set instance(ggmlimpl.ggml_opt_params params) => ptr.ref = params;

  ggmlimpl.ggml_opt_params get instance => ptr.ref;

  static int get size => sizeOf<ggmlimpl.ggml_opt_params>();

  void free() => ffi.calloc.free(ptr);
}
