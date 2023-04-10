/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

enum GgmlType {
  none(-1),
  q40(ggmlimpl.ggml_type.GGML_TYPE_Q4_0),
  q41(ggmlimpl.ggml_type.GGML_TYPE_Q4_1),
  i8(ggmlimpl.ggml_type.GGML_TYPE_I8),
  i16(ggmlimpl.ggml_type.GGML_TYPE_I16),
  i32(ggmlimpl.ggml_type.GGML_TYPE_I32),
  f16(ggmlimpl.ggml_type.GGML_TYPE_F16),
  f32(ggmlimpl.ggml_type.GGML_TYPE_F32),
  count(ggmlimpl.ggml_type.GGML_TYPE_COUNT);

  static final Map<int, GgmlType> byCode = {};

  static GgmlType type(int type) {
    if (byCode.isEmpty) {
      for (final type in GgmlType.values) {
        byCode[type.code] = type;
      }
    }

    final ret = byCode.containsKey(type) ? byCode[type] : GgmlType.none;
    return ret!;
  }

  @override
  String toString() {
    return "$name($code)";
  }

  final int code;

  const GgmlType(this.code);
}
