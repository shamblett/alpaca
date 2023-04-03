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
class Utils {
  /// Default uses the supplied dynamic library
  Utils() {
    _impl = utilsimpl.UtilsImpl(DynamicLibrary.open('library/libutils.so'));
  }

  /// Specify the library and path
  Utils.fromLib(String libPath) {
    _impl = utilsimpl.UtilsImpl(DynamicLibrary.open(libPath));
  }

  // The MRAA Implementation class
  late utilsimpl.UtilsImpl _impl;

}
