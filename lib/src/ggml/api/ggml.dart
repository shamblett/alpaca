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
  void timeInit() => _impl.ggml_time_init();

  /// int64_t ggml_time_ms(void);
  int timeMs() => _impl.ggml_time_ms();

  /// int64_t ggml_time_us(void);
  int timeUs() => _impl.ggml_time_us();

  /// int64_t ggml_cycles(void);
  int cycles() => _impl.ggml_cycles();

  /// int64_t ggml_cycles_per_ms(void);
  int cyclesPerMs() => _impl.ggml_cycles_per_ms();

  /// void ggml_print_object(const struct ggml_object * obj);
  void printObject(GgmlObject obj) => _impl.ggml_print_object(obj);

  /// void ggml_print_objects(const struct ggml_context * ctx);
  void printObjects(GgmlContext ctx) => _impl.ggml_print_objects(ctx);

  /// int ggml_nelements(const struct ggml_tensor * tensor);
  int nElements(GgmlTensor tensor) {
    final ptr = ffi.calloc<GgmlTensor>();
    ptr.ref = tensor;
    final ret = _impl.ggml_nelements(ptr);
    ffi.calloc.free(ptr);
    return ret;
  }

  /// size_t ggml_nbytes   (const struct ggml_tensor * tensor);
  int nBytes(GgmlTensor tensor) {
    final ptr = ffi.calloc<GgmlTensor>();
    ptr.ref = tensor;
    final ret = _impl.ggml_nbytes(ptr);
    ffi.calloc.free(ptr);
    return ret;
  }

  /// int ggml_blck_size (enum ggml_type type);
  // int blockSize(GgmlType theType) => _impl.ggml_blck_size(theType.);
}
