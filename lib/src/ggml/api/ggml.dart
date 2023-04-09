/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

/// The Alpaca GGML interface.
///
///  For a full description of the methods in this class please see its
///  corresponding C header file https://github.com/antimatter15/alpaca.cpp/blob/master/ggml.h
///
class Ggml {
  Ggml() {
    _impl = ggmlimpl.GgmlImpl(
        DynamicLibrary.open('lib/src/ggml/implementation/library/libggml.so'));
  }

  /// Specify the library and path
  Ggml.fromLib(String libPath) {
    _impl = ggmlimpl.GgmlImpl(DynamicLibrary.open(libPath));
  }

  // The GGML Implementation class
  late ggmlimpl.GgmlImpl _impl;

  /// void ggml_time_init(void);
  void ggmTimeInit() => _impl.ggml_time_init();

  /// int64_t ggml_time_ms(void);
  int ggmlTimeMs() => _impl.ggml_time_ms();

  /// int64_t ggml_time_us(void);
  int ggmlTimeUs() => _impl.ggml_time_us();

  /// int64_t ggml_cycles(void);
  int ggmlCycles() => _impl.ggml_cycles();

  /// int64_t ggml_cycles_per_ms(void);
  int ggmlCyclesPerMs() => _impl.ggml_cycles_per_ms();

  /// void ggml_print_object(const struct ggml_object * obj);
  void ggmlPrintObject(GgmlObject obj) => _impl.ggml_print_object(obj);
}
