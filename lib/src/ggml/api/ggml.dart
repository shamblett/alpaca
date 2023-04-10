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

  /// int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
  int getI321d(GgmlTensor tensor, int i) =>
      _impl.ggml_get_i32_1d(tensor.ptr, i);

  /// void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);
  void setI321d(GgmlTensor tensor, int i, int value) =>
      _impl.ggml_set_i32_1d(tensor.ptr, i, value);

  /// float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
  double getF321d(GgmlTensor tensor, int i) =>
      _impl.ggml_get_f32_1d(tensor.ptr, i);

  /// void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);
  void setF321d(GgmlTensor tensor, int i, double value) =>
      _impl.ggml_set_f32_1d(tensor.ptr, i, value);

  /// void * ggml_get_data    (const struct ggml_tensor * tensor);
  Pointer<void> getData(GgmlTensor tensor) => _impl.ggml_get_data(tensor.ptr);

  /// float * ggml_get_data_f32(const struct ggml_tensor * tensor);
  Pointer<Float> getDataF32(GgmlTensor tensor) =>
      _impl.ggml_get_data_f32(tensor.ptr);

  /// void ggml_set_param(struct ggml_context * ctx, struct ggml_tensor * tensor);
  void setParam(GgmlContext ctx, GgmlTensor tensor) =>
      _impl.ggml_set_param(ctx, tensor.ptr);

  /// void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
  void buildForwardExpand(GgmlCGraph cGraph, GgmlTensor tensor) =>
      _impl.ggml_build_forward_expand(cGraph.ptr, tensor.ptr);

  /// struct ggml_cgraph ggml_build_forward (struct ggml_tensor * tensor);
  GgmlCGraph buildForward(GgmlTensor tensor) {
    final cGraph = _impl.ggml_build_forward(tensor.ptr);
    return GgmlCGraph()..instance = cGraph;
  }

  /// struct ggml_cgraph ggml_build_backward(struct ggml_context * ctx, struct ggml_cgraph * gf, bool keep);
  GgmlCGraph buildBackward(GgmlContext ctx, GgmlCGraph gf, bool keep) {
    final cGraph = _impl.ggml_build_backward(ctx, gf.ptr, keep);
    return GgmlCGraph()..instance = cGraph;
  }

  /// void ggml_graph_compute(struct ggml_context * ctx, struct ggml_cgraph * cgraph);
  void graphCompute(GgmlContext ctx, GgmlCGraph gf) =>
      _impl.ggml_graph_compute(ctx, gf.ptr);

  /// void ggml_graph_reset  (struct ggml_cgraph * cgraph);
  void graphReset(GgmlCGraph cGraph) => _impl.ggml_graph_reset(cGraph.ptr);

  /// void ggml_graph_print(const struct ggml_cgraph * cgraph);
  void graphPrint(GgmlCGraph cGraph) => _impl.ggml_graph_print(cGraph.ptr);

  /// void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);
  void graphDumpDot(GgmlCGraph gb, GgmlCGraph gf, String filename) =>
      _impl.ggml_graph_dump_dot(
          gb.ptr, gf.ptr, filename.toNativeUtf8().cast<Char>());

  /// struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type);
  GgmlOptParams optDefaultParams(GgmlOptType type) {
    final optParams = _impl.ggml_opt_default_params(type.code);
    return GgmlOptParams()..instance = optParams;
  }

  /// enum ggml_opt_result ggml_opt(struct ggml_context * ctx,struct ggml_opt_params params,
  /// struct ggml_tensor * f);
  GgmlOptResult opt(GgmlContext ctx, GgmlOptParams params, GgmlTensor f) =>
      GgmlOptResult.opt(_impl.ggml_opt(ctx, params.instance, f.ptr));

  /// int ggml_cpu_has_avx(void);
  int cpuHasAvx() => _impl.ggml_cpu_has_avx();

  /// int ggml_cpu_has_avx2(void);
  int cpuHasAvx2() => _impl.ggml_cpu_has_avx2();

  /// int ggml_cpu_has_avx512(void);
  int cpuHasAvx512() => _impl.ggml_cpu_has_avx512();

  /// int ggml_cpu_has_fma(void);
  int cpuHasFma() => _impl.ggml_cpu_has_fma();

  /// int ggml_cpu_has_neon(void);
  int cpuHasNeon() => _impl.ggml_cpu_has_neon();

  /// int ggml_cpu_has_arm_fma(void);
  int cpuHasArmFma() => _impl.ggml_cpu_has_arm_fma();

  /// int ggml_cpu_has_f16c(void);
  int cpuHasF16c() => _impl.ggml_cpu_has_f16c();

  /// int ggml_cpu_has_avx(void);
  int cpuHasFp16Va() => _impl.ggml_cpu_has_fp16_va();

  /// int ggml_cpu_has_wasm_simd(void);
  int cpuHasWasmSimd() => _impl.ggml_cpu_has_wasm_simd();

  /// int ggml_cpu_has_blas(void);
  int cpuHasBlas() => _impl.ggml_cpu_has_blas();

  /// int ggml_cpu_has_sse3(void);
  int cpuHasSse3() => _impl.ggml_cpu_has_sse3();

  /// int ggml_cpu_has_vsx(void);
  int cpuHasVsx() => _impl.ggml_cpu_has_vsx();
}
