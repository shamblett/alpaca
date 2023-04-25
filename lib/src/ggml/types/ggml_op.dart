/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

enum GgmlOp {
  none(0),
  dup(ggmlimpl.ggml_op.GGML_OP_DUP),
  add(ggmlimpl.ggml_op.GGML_OP_ADD),
  sub(ggmlimpl.ggml_op.GGML_OP_SUB),
  mul(ggmlimpl.ggml_op.GGML_OP_MUL),
  div(ggmlimpl.ggml_op.GGML_OP_DIV),
  sqr(ggmlimpl.ggml_op.GGML_OP_SQR),
  sqrt(ggmlimpl.ggml_op.GGML_OP_SQRT),
  sum(ggmlimpl.ggml_op.GGML_OP_SUM),
  mean(ggmlimpl.ggml_op.GGML_OP_MEAN),
  repeat(ggmlimpl.ggml_op.GGML_OP_REPEAT),
  abs(ggmlimpl.ggml_op.GGML_OP_ABS),
  sgn(ggmlimpl.ggml_op.GGML_OP_SGN),
  neg(ggmlimpl.ggml_op.GGML_OP_NEG),
  step(ggmlimpl.ggml_op.GGML_OP_STEP),
  relu(ggmlimpl.ggml_op.GGML_OP_RELU),
  gelu(ggmlimpl.ggml_op.GGML_OP_GELU),
  silu(ggmlimpl.ggml_op.GGML_OP_SILU),
  norm(ggmlimpl.ggml_op.GGML_OP_NORM),
  rmsNorm(ggmlimpl.ggml_op.GGML_OP_RMS_NORM),
  mulMat(ggmlimpl.ggml_op.GGML_OP_MUL_MAT),
  scale(ggmlimpl.ggml_op.GGML_OP_SCALE),
  cpy(ggmlimpl.ggml_op.GGML_OP_CPY),
  reshape(ggmlimpl.ggml_op.GGML_OP_RESHAPE),
  view(ggmlimpl.ggml_op.GGML_OP_VIEW),
  permute(ggmlimpl.ggml_op.GGML_OP_PERMUTE),
  transpose(ggmlimpl.ggml_op.GGML_OP_TRANSPOSE),
  getRows(ggmlimpl.ggml_op.GGML_OP_GET_ROWS),
  diagMaskInf(ggmlimpl.ggml_op.GGML_OP_DIAG_MASK_INF),
  softMax(ggmlimpl.ggml_op.GGML_OP_SOFT_MAX),
  rope(ggmlimpl.ggml_op.GGML_OP_ROPE),
  conv1d1s(ggmlimpl.ggml_op.GGML_OP_CONV_1D_1S),
  conv1d2s(ggmlimpl.ggml_op.GGML_OP_CONV_1D_2S),
  flashAttn(ggmlimpl.ggml_op.GGML_OP_FLASH_ATTN),
  flashFf(ggmlimpl.ggml_op.GGML_OP_FLASH_FF),
  count(ggmlimpl.ggml_op.GGML_OP_COUNT);

  static final Map<int, GgmlOp> byCode = {};

  static GgmlOp op(int type) {
    if (byCode.isEmpty) {
      for (final type in GgmlOp.values) {
        byCode[type.code] = type;
      }
    }

    final ret = byCode.containsKey(type) ? byCode[type] : GgmlOp.none;
    return ret!;
  }

  @override
  String toString() {
    return "$name($code)";
  }

  final int code;

  const GgmlOp(this.code);
}
