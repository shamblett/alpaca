/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

import 'dart:io';

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

  final vocab = AlpacaGptVocab();
  final model = AlpacaLlamaModel();

  // Load the model
  {
    final modelPath = '${Directory.current.path}/model/${params.model}';
    print('Model path is $modelPath');
    final tStartUs = ggml.timeUs();
    if (!AlpacaChat.llamaModelLoad(
        modelPath, model, vocab, params.nCtx, ggml)) {
      print('AlpacaChat:: failed to load model from $modelPath\n');
      return 1;
    }

    final tLoadUs = ggml.timeUs() - tStartUs;
  }

  // Print system information
  {
    final numProcessors = Platform.numberOfProcessors;
    print('');
    print(
        'AlpacaChat:: System_info: n_threads = ${params.nThreads} / $numProcessors | ${AlpacaChat.llamaPrintSystemInfo(ggml)}');
  }

  int nPast = 0;

  int tSampleUs = 0;
  int tPredictUs = 0;

  final logits = <double>[];

  // Tokenize the prompt
  final embdInp = <Id?>[];

  final instructInp = AlpacaUtils.llamaTokenize(
      vocab,
      ' Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n',
      true);
  final promptInp =
      AlpacaUtils.llamaTokenize(vocab, '### Instruction:\n\n', true);
  final responseInp =
      AlpacaUtils.llamaTokenize(vocab, '### Response:\n\n', false);
  embdInp.addAll(instructInp);

  if (params.prompt.isNotEmpty) {
    final paramInp = AlpacaUtils.llamaTokenize(vocab, params.prompt, true);
    embdInp.addAll(promptInp);
    embdInp.addAll(paramInp);
    embdInp.addAll(responseInp);
  }

  if (params.interactive) {
    print('AlpacaChat:: interactive mode on');
  }
  print(
      'AlpacaChat:: sampling parameters: temp = ${params.temp}, top_k = ${params.topK}, top_p = ${params.topP}, repeat_last_n = ${params.repeatLastN}, repeat_penalty = ${params.repeatPenalty}');
  print('');
  print('');

  final embd = <Id>[];

  // Determine the required inference memory per token:
  int memPerToken = 0;
  //llama_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

  return 0;
}
