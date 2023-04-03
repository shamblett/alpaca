/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

part of alpaca;

///
/// The main Utils interface.
///
class AlpacaUtils {
  /// Default uses the supplied dynamic library
  AlpacaUtils() {
    _impl = utilsimpl.UtilsImpl(DynamicLibrary.open('lib/src/utils/implementation/library/libutils.so'));
  }

  /// Specify the library and path
  AlpacaUtils.fromLib(String libPath) {
    _impl = utilsimpl.UtilsImpl(DynamicLibrary.open(libPath));
  }

  // The MRAA Implementation class
  late utilsimpl.UtilsImpl _impl;
}
