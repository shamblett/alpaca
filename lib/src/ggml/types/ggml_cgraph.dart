/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

class GgmlCGraph {
  // Finalized to free the resource should the user not do so
  Pointer<ggmlimpl.ggml_cgraph> ptr = ffi.calloc<ggmlimpl.ggml_cgraph>();

  ggmlimpl.ggml_cgraph get instance => ptr.ref;

  set instance(ggmlimpl.ggml_cgraph cGraph) => ptr.ref = cGraph;

  static int get size => sizeOf<ggmlimpl.ggml_cgraph>();

  void free() => ffi.calloc.free(ptr);

  @override
  toString() => 'Nodes = ${instance.n_nodes}, leafs = ${instance.n_leafs}';
}
