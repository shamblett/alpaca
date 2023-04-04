/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of alpaca;

/// The Alpaca GGML interface.

class AlpacaGgml {
  AlpacaGgml() {
    _impl =
        ggmlimpl.GgmlImpl(DynamicLibrary.open('lib/src/ggml/implementation/library/libggml.so'));
  }

  /// Specify the library and path
  AlpacaGgml.fromLib(String libPath) {
    _impl = ggmlimpl.GgmlImpl(DynamicLibrary.open(libPath));
  }

  // The GGML Implementation class
  late ggmlimpl.GgmlImpl _impl;
}
