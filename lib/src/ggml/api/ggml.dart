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

  /// struct ggml_tensor * ggml_new_tensor(struct ggml_context * ctx, enum ggml_type type,
  /// int n_dims, const int *ne);
  GgmlTensor newTensor(
      GgmlContext ctx, GgmlType type, int nDims, Pointer<Int> ne) {
    final ptr = _impl.ggml_new_tensor(ctx, type.code, nDims, ne);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_new_tensor_1d(struct ggml_context * ctx,
  /// enum ggml_type type, int ne0);
  GgmlTensor newTensor1D(GgmlContext ctx, GgmlType type, int ne0) {
    final ptr = _impl.ggml_new_tensor_1d(ctx, type.code, ne0);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_new_tensor_2d(struct ggml_context * ctx,
  /// enum ggml_type type, int ne0, int ne1);
  GgmlTensor newTensor2D(GgmlContext ctx, GgmlType type, int ne0, int ne1) {
    final ptr = _impl.ggml_new_tensor_2d(ctx, type.code, ne0, ne1);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_new_tensor_3d(struct ggml_context * ctx,
  /// enum ggml_type type, int ne0, int ne1, int ne2);
  GgmlTensor newTensor3D(
      GgmlContext ctx, GgmlType type, int ne0, int ne1, int ne2) {
    final ptr = _impl.ggml_new_tensor_3d(ctx, type.code, ne0, ne1, ne2);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_new_tensor_4d(struct ggml_context * ctx,
  /// enum ggml_type type, int ne0, int ne1, int ne2, int ne3);
  GgmlTensor newTensor4D(
      GgmlContext ctx, GgmlType type, int ne0, int ne1, int ne2, int ne3) {
    final ptr = _impl.ggml_new_tensor_4d(ctx, type.code, ne0, ne1, ne2, ne3);
    return GgmlTensor()..ptr = ptr;
  }

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

  /// struct ggml_tensor * ggml_dup(struct ggml_context * ctx,struct ggml_tensor  * a);
  GgmlTensor dup(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_dup(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_add(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor add(GgmlContext ctx, GgmlTensor a, GgmlTensor b) {
    final ptr = _impl.ggml_add(ctx, a.ptr, b.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_sub(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor sub(GgmlContext ctx, GgmlTensor a, GgmlTensor b) {
    final ptr = _impl.ggml_sub(ctx, a.ptr, b.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_mul(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor mul(GgmlContext ctx, GgmlTensor a, GgmlTensor b) {
    final ptr = _impl.ggml_mul(ctx, a.ptr, b.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_div(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor div(GgmlContext ctx, GgmlTensor a, GgmlTensor b) {
    final ptr = _impl.ggml_div(ctx, a.ptr, b.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_sqr(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor sqr(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_sqr(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_sqrt(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor sqrt(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_sqrt(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_sum(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor sum(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_sum(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_mean(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor mean(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_mean(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_repeat(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor repeat(GgmlContext ctx, GgmlTensor a, GgmlTensor b) {
    final ptr = _impl.ggml_repeat(ctx, a.ptr, b.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_abs(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor abs(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_abs(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_sgn(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor sgn(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_sgn(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_neg(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor neg(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_neg(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_step(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor step(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_step(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_relu(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor relu(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_relu(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_gelu(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor gelu(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_gelu(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_silu(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor silu(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_silu(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_norm(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor norm(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_norm(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_rms_norm(struct ggml_context * ctx,
  /// struct ggml_tensor  * a);
  GgmlTensor rmsNorm(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_rms_norm(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_mul_mat(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor mulMat(GgmlContext ctx, GgmlTensor a, GgmlTensor b) {
    final ptr = _impl.ggml_mul_mat(ctx, a.ptr, b.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_scale(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor scale(GgmlContext ctx, GgmlTensor a, GgmlTensor b) {
    final ptr = _impl.ggml_scale(ctx, a.ptr, b.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_cpy(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor cpy(GgmlContext ctx, GgmlTensor a, GgmlTensor b) {
    final ptr = _impl.ggml_cpy(ctx, a.ptr, b.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_reshape(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor reshape(GgmlContext ctx, GgmlTensor a, GgmlTensor b) {
    final ptr = _impl.ggml_reshape(ctx, a.ptr, b.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_reshape_2d(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, int ne0, int ne1);
  GgmlTensor reshape2D(GgmlContext ctx, GgmlTensor a, int ne0, int ne1) {
    final ptr = _impl.ggml_reshape_2d(ctx, a.ptr, ne0, ne1);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_reshape_3d(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, int ne0, int ne1, int ne2);
  GgmlTensor reshape3D(
      GgmlContext ctx, GgmlTensor a, int ne0, int ne1, int ne2) {
    final ptr = _impl.ggml_reshape_3d(ctx, a.ptr, ne0, ne1, ne2);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_view_1d(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, int ne0, int offset);
  GgmlTensor view1D(GgmlContext ctx, GgmlTensor a, int ne0, int offset) {
    final ptr = _impl.ggml_view_1d(ctx, a.ptr, ne0, offset);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_view_2d(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, int ne0, int ne1, int nb1, int offset);
  GgmlTensor view2D(
      GgmlContext ctx, GgmlTensor a, int ne0, int ne1, int nb1, int offset) {
    final ptr = _impl.ggml_view_2d(ctx, a.ptr, ne0, ne1, nb1, offset);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_permute(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, int axis0, int axis1, int axis2, int axis3);
  GgmlTensor transpose(GgmlContext ctx, GgmlTensor a, int axis0, int axis1,
      int axis2, int axis3) {
    final ptr = _impl.ggml_permute(ctx, a.ptr, axis0, axis1, axis2, axis3);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_get_rows(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor getRows(GgmlContext ctx, GgmlTensor a, GgmlTensor b) {
    final ptr = _impl.ggml_get_rows(ctx, a.ptr, b.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_diag_mask_inf(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, int n_past);
  GgmlTensor diagMaskInf(GgmlContext ctx, GgmlTensor a, int nPast) {
    final ptr = _impl.ggml_diag_mask_inf(ctx, a.ptr, nPast);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_soft_max(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor softMax(GgmlContext ctx, GgmlTensor a) {
    final ptr = _impl.ggml_soft_max(ctx, a.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_rope(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, int n_past, int n_dims, int mode);
  GgmlTensor rope(
      GgmlContext ctx, GgmlTensor a, int nPast, int nDims, int mode) {
    final ptr = _impl.ggml_rope(ctx, a.ptr, nPast, nDims, mode);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_conv_1d_1s(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor conv1D1S(GgmlContext ctx, GgmlTensor a, GgmlTensor b) {
    final ptr = _impl.ggml_conv_1d_1s(ctx, a.ptr, b.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_conv_1d_2s(struct ggml_context * ctx,
  /// struct ggml_tensor  * a, struct ggml_tensor  * b);
  GgmlTensor conv1D2S(GgmlContext ctx, GgmlTensor a, GgmlTensor b) {
    final ptr = _impl.ggml_conv_1d_2s(ctx, a.ptr, b.ptr);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_flash_attn(struct ggml_context * ctx,
  /// struct ggml_tensor * q, struct ggml_tensor * k, struct ggml_tensor * v,
  /// bool masked);
  GgmlTensor flashAttn(
      GgmlContext ctx, GgmlTensor q, GgmlTensor k, GgmlTensor v, bool masked) {
    final ptr = _impl.ggml_flash_attn(ctx, q.ptr, k.ptr, v.ptr, masked);
    return GgmlTensor()..ptr = ptr;
  }

  /// struct ggml_tensor * ggml_flash_ff(struct ggml_context * ctx,
  /// struct ggml_tensor * a, struct ggml_tensor * b0, struct ggml_tensor * b1,
  /// struct ggml_tensor * c0, struct ggml_tensor * c1);
  GgmlTensor flashFf(GgmlContext ctx, GgmlTensor a, GgmlTensor b0,
      GgmlTensor b1, GgmlTensor c0, GgmlTensor c1) {
    final ptr = _impl.ggml_flash_ff(ctx, a.ptr, b0.ptr, b1.ptr, c0.ptr, c1.ptr);
    return GgmlTensor()..ptr = ptr;
  }

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
