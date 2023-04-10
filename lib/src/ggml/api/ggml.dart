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
/// The methods are implemented in the same order as that found in the C header file.
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
  int nElements(GgmlTensor tensor) => _impl.ggml_nelements(tensor.ptr);

  /// size_t ggml_nbytes   (const struct ggml_tensor * tensor);
  int nBytes(GgmlTensor tensor) => _impl.ggml_nbytes(tensor.ptr);

  /// int ggml_blck_size (enum ggml_type type);
  int blockSize(GgmlType theType) => _impl.ggml_blck_size(theType.code);

  /// size_t ggml_type_size (enum ggml_type type); // size in bytes for all elements in a block
  int typeSize(GgmlType theType) => _impl.ggml_type_size(theType.code);

  /// float ggml_type_sizef(enum ggml_type type); // ggml_type_size()/ggml_blck_size() as float
  double typeSizeF(GgmlType theType) => _impl.ggml_type_sizef(theType.code);

  /// size_t ggml_element_size(const struct ggml_tensor * tensor);
  int elementSize(GgmlTensor tensor) => _impl.ggml_element_size(tensor.ptr);

  /// struct ggml_context * ggml_init(struct ggml_init_params params);
  GgmlContext init(GgmlInitParams params) => _impl.ggml_init(params.instance);

  /// void ggml_free(struct ggml_context * ctx);
  void free(GgmlContext ctx) => _impl.ggml_free(ctx);

  /// size_t ggml_used_mem(const struct ggml_context * ctx);
  int usedMem(GgmlContext ctx) => _impl.ggml_used_mem(ctx);

  /// size_t ggml_set_scratch(struct ggml_context * ctx, struct ggml_scratch scratch);
  int setScratch(GgmlContext ctx, GgmlScratch scratch) =>
      _impl.ggml_set_scratch(ctx, scratch.instance);

  /// struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
  GgmlTensor newI32(GgmlContext ctx, int value) {
    final ptr = _impl.ggml_new_i32(ctx, value);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);
  GgmlTensor newF32(GgmlContext ctx, double value) {
    final ptr = _impl.ggml_new_f32(ctx, value);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
  GgmlTensor dupTensor(GgmlContext ctx, GgmlTensor tensor) {
    final ptr = _impl.ggml_dup_tensor(ctx, tensor.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, const struct ggml_tensor * src);
  GgmlTensor viewTensor(GgmlContext ctx, GgmlTensor tensor) {
    final ptr = _impl.ggml_view_tensor(ctx, tensor.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
  GgmlTensor setZero(GgmlTensor tensor) {
    final ptr = _impl.ggml_set_zero(tensor.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
  GgmlTensor setI32(GgmlTensor tensor, int value) {
    final ptr = _impl.ggml_set_i32(tensor.ptr, value);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);
  GgmlTensor setF32(GgmlTensor tensor, double value) {
    final ptr = _impl.ggml_set_f32(tensor.ptr, value);
    return GgmlTensor()..ptr = ptr;
  }
}
