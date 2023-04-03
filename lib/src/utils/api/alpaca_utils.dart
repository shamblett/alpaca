/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

part of alpaca;

///
/// The main Utility library interface.
///
class AlpacaUtils {
  /// Default uses the supplied dynamic library
  AlpacaUtils() {
    _impl = utilsimpl.UtilsImpl(DynamicLibrary.open(
        'lib/src/utils/implementation/library/libutils.so'));
  }

  /// Specify the library and path
  AlpacaUtils.fromLib(String libPath) {
    _impl = utilsimpl.UtilsImpl(DynamicLibrary.open(libPath));
  }

  // The Utility Implementation class
  late utilsimpl.UtilsImpl _impl;

  /// API methods, lack of comment reflects lack in the C header file.

  bool gptParamsParse(int argc, List<String> argv, AlpacaGptParams arg2) {
    final ret = _impl.gpt_params_parse(
      argc,
      _strListToCharPointer(argv),
      arg2,
    );

    return ret != 0;
  }


  Pointer<Pointer<Char>> _strListToCharPointer(List<String> strings) {
    // Gat the strings as UTF8
    List<Pointer<ffi.Utf8>> utf8PointerList =
        strings.map((str) => str.toNativeUtf8()).toList();

    // Cast to Char
    final charPointerList = utf8PointerList.cast<Pointer<Char>>();

    final Pointer<Pointer<Char>> pointerPointer =
        ffi.malloc.allocate(charPointerList.length);

    // Return the strings
    strings.asMap().forEach((index, utf) {
      pointerPointer[index] = charPointerList[index];
    });

    return pointerPointer;
  }
}
