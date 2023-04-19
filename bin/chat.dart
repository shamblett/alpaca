/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

import 'dart:io';
import 'dart:math';

import 'package:fixnum/fixnum.dart';
import 'package:mt19937/mt19937.dart';
import 'package:alpaca/alpaca.dart';
import 'package:alpaca/src/ggml/ggml.dart';

bool isInteracting = false;

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
  AlpacaChat.llamaEval(
      model, params.nThreads, 0, [0, 1, 2, 3], logits, memPerToken);

  int lastNSize = params.repeatLastN;
  final lastNTokens = List<Id>.filled(lastNSize, 0);

  if (params.interactive) {
    print('AlpacaChat:: == Running in chat mode. ==');
    print('AlpacaChat::  - Press Ctrl+C to interject at any time.');

    // We may want to slide the input window along with the context, but for now we restrict to the context length
    int remainingTokens = model.hParams!.nCtx - embdInp.length;
    int inputConsumed = 0;
    bool inputNoEcho = true;

    // Prompt user immediately after the starting prompt has been loaded
    if (params.interactiveStart) {
      isInteracting = true;
    }

    // set the color for the prompt which will be output initially
    if (params.useColor) {
      print(ansiColorYellow);
    }

    while (remainingTokens > 0) {
      // Predict
      if (embd.isNotEmpty) {
        final tStartUs = ggml.timeUs();

        if (!AlpacaChat.llamaEval(
            model, params.nThreads, nPast, embd, logits, memPerToken)) {
          print('AlpacaChat:: Failed to predict');
          return 1;
        }

        tPredictUs += ggml.timeUs() - tStartUs;
      }

      nPast += embd.length;
      embd.clear();

      if (embdInp.length <= inputConsumed && !isInteracting) {
        // out of user input, sample next token
        final topK = params.topK;
        final topP = params.topP;
        final temp = params.temp;
        final repeatPenalty = params.repeatPenalty;

        final nVocab = model.hParams?.nVocab;

        Id id = 0;

        {
          final tStartSampleUs = ggml.timeUs();

          final rand = Random();
          id = AlpacaUtils.llamaSampleTopPTopK(vocab, logits, lastNTokens,
              repeatPenalty, topK, topP, temp, rand);

          lastNTokens.clear();
          lastNTokens.add(id);

          tSampleUs += ggml.timeUs() - tStartSampleUs;
        }

        // Add it to the context
        embd.add(id);

        // Echo this to console
        inputNoEcho = false;

        // Decrement remaining sampling budget
        --remainingTokens;
      } else {
        // Some user input remains from prompt or interaction, forward it to processing
        while (embdInp.length > inputConsumed) {
          embd.add(embdInp[inputConsumed]!);
          lastNTokens.clear();
          lastNTokens.add(embdInp[inputConsumed]!);
          ++inputConsumed;
          if (embd.length > params.nBatch) {
            break;
          }
        }

        // Reset color to default if we there is no pending user input
        if (!inputNoEcho &&
            params.useColor &&
            embdInp.length == inputConsumed) {
          print(ansiColorReset);
        }
      }

      // Display text
      if (!inputNoEcho) {
        for (final id in embd) {
          stdout.write('${vocab.idToToken[id]}');
        }
        stdout.flush();
      }

      // In interactive mode, and not currently processing queued inputs;
      // check if we should prompt the user for more
      if (params.interactive && embdInp.length <= inputConsumed) {
        if (isInteracting) {
          inputConsumed = embdInp.length;
          embdInp.addAll(promptInp);
          print('>');

          // Currently being interactive
          bool anotherLine = true;
          while (anotherLine) {
            if (params.useColor) {
              print(ansiBold);
              print(ansiColorGreen);
            }
            var line = stdin.readLineSync();
            if (params.useColor) {
              print(ansiColorReset);
            }
            line != null && line.endsWith('\\')
                ? anotherLine = true
                : anotherLine = false;

            final lineInp = AlpacaUtils.llamaTokenize(vocab, line!, false);
            embdInp.addAll(lineInp);
            embdInp.addAll(responseInp);

            remainingTokens -=
                promptInp.length + lineInp.length + responseInp.length;

            inputNoEcho = true;
          }

          isInteracting = false;
        }
      }
      // End of text token
      if (embd.last == 2) {
        if (params.interactive) {
          isInteracting = true;
          continue;
        } else {
          print('');
          print(' [end of text]');
          break;
        }
      }
    }
  }

  // Report timing
  {
    final tMainEndUs = ggml.timeUs();

    print('');
    print('');
    print('AlpacaChat:: mem per token = $memPerToken bytes');
    print('AlpacaChat:: load time = ${tLoadUs / 1000.0} ms');
    print('AlpacaChat:: sample time = ${tSampleUs / 1000.0} ms');
    print(
        'AlpacaChat:: predict time = ${tPredictUs / 1000} ms / ${tPredictUs / 1000.0 / nPast} ms per token');
    print(
        'AlpacaChat:: total time = ${(tMainEndUs - tMainStartUs) / 1000.0} ms');
  }

  ggml.free(model.ctx!);

  if (params.useColor) {
    print(ansiColorReset);
  }

  return 0;
}
