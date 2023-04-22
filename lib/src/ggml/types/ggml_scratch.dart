/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

class GgmlScratch {
  // Finalized to free the resource should the user not do so
  Pointer<ggmlimpl.ggml_scratch> ptr = ffi.calloc<ggmlimpl.ggml_scratch>();

  ggmlimpl.ggml_scratch get instance => ptr.ref;

  static int get size => sizeOf<ggmlimpl.ggml_scratch>();

  void free() => ffi.calloc.free(ptr);
}
