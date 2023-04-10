/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

enum GgmlOptResult {
  none(-1),
  ok(ggmlimpl.ggml_opt_result.GGML_OPT_OK),
  didNotConverge(ggmlimpl.ggml_opt_result.GGML_OPT_DID_NOT_CONVERGE),
  noContext(ggmlimpl.ggml_opt_result.GGML_OPT_NO_CONTEXT),
  invalidWolfe(ggmlimpl.ggml_opt_result.GGML_OPT_INVALID_WOLFE),
  fail(ggmlimpl.ggml_opt_result.GGML_OPT_FAIL),
  lineSearchFail(ggmlimpl.ggml_opt_result.GGML_LINESEARCH_FAIL),
  lineSearchMinimumStep(ggmlimpl.ggml_opt_result.GGML_LINESEARCH_MINIMUM_STEP),
  lineSearchMaximumStep(ggmlimpl.ggml_opt_result.GGML_LINESEARCH_MAXIMUM_STEP),
  lineSearchMaximumIterations(
      ggmlimpl.ggml_opt_result.GGML_LINESEARCH_MAXIMUM_ITERATIONS),
  lineSearchInvalidParameters(
      ggmlimpl.ggml_opt_result.GGML_LINESEARCH_INVALID_PARAMETERS);

  static final Map<int, GgmlOptResult> byCode = {};

  static GgmlOptResult opt(int type) {
    if (byCode.isEmpty) {
      for (final type in GgmlOptResult.values) {
        byCode[type.code] = type;
      }
    }

    final ret = byCode.containsKey(type) ? byCode[type] : GgmlOptResult.none;
    return ret!;
  }

  @override
  String toString() {
    return "$name($code)";
  }

  final int code;

  const GgmlOptResult(this.code);
}
