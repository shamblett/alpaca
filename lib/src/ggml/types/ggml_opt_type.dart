/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

enum GgmlOptType {
  none(-1),
  adam(ggmlimpl.ggml_opt_type.GGML_OPT_ADAM),
  lbfgs(ggmlimpl.ggml_opt_type.GGML_OPT_LBFGS);

  static final Map<int, GgmlOptType> byCode = {};

  static GgmlOptType type(int type) {
    if (byCode.isEmpty) {
      for (final type in GgmlOptType.values) {
        byCode[type.code] = type;
      }
    }

    final ret = byCode.containsKey(type) ? byCode[type] : GgmlOptType.none;
    return ret!;
  }

  @override
  String toString() {
    return "$name($code)";
  }

  final int code;

  const GgmlOptType(this.code);
}
