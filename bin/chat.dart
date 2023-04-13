/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */


import 'package:fixnum/fixnum.dart';
import 'package:mt19937/mt19937.dart';
import 'package:alpaca/alpaca.dart';
import 'package:alpaca/src/ggml/ggml.dart';

///
/// Main chat application
///
int main(List<String> argv) {
  final ggml = Ggml();

  ggml.timeInit();
  int tMainStartUs = ggml.timeUs();
  final params = AlpacaGptParams();

  if (AlpacaUtils.gptParamsParse(argv.length, argv, params) == false) {
    return 1;
  }

  if (params.seed < 0) {
    params.seed = DateTime.now().millisecondsSinceEpoch * 1000;
  }

  print('seed = ${params.seed}\n');

  var rng = MersenneTwisterEngine.w32()..init(Int64(params.seed));

  int tLoadUs = 0;

  final AlpacaGptVocab vocab;
  final AlpacaLlamaModel model;



  return 0;
}
