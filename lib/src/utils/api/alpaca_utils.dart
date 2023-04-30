/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

part of alpaca;

///
/// The main Utility library interface.
///
class AlpacaUtils {
  AlpacaUtils();

  static const maxTokenLen = 18;

  /// API methods, lack of comment reflects lack of same in the implementation files.

  static bool gptParamsParse(
      int argc, List<String> argv, AlpacaGptParams params) {
    for (int i = 1; i < argc; i++) {
      String arg = argv[i];

      if (arg == "-s" || arg == "--seed") {
        params.seed = int.parse(argv[++i]);
      } else if (arg == "-t" || arg == "--threads") {
        params.nThreads = int.parse(argv[++i]);
      } else if (arg == "-p" || arg == "--prompt") {
        params.interactive = false;
        params.interactiveStart = false;
        params.useColor = false;

        params.prompt = argv[++i];
      } else if (arg == "-f" || arg == "--file") {
        params.interactive = false;
        params.interactiveStart = false;
        params.useColor = false;
      } else if (arg == "-n" || arg == "--n_predict") {
        params.nPredict = int.parse(argv[++i]);
      } else if (arg == "--top_k") {
        params.topK = int.parse(argv[++i]);
      } else if (arg == "-c" || arg == "--ctx_size") {
        params.nCtx = int.parse(argv[++i]);
      } else if (arg == "--top_p") {
        params.topP = double.parse(argv[++i]);
      } else if (arg == "--temp") {
        params.temp = double.parse(argv[++i]);
      } else if (arg == "--repeat_last_n") {
        params.repeatLastN = int.parse(argv[++i]);
      } else if (arg == "--repeat_penalty") {
        params.repeatPenalty = double.parse(argv[++i]);
      } else if (arg == "-b" || arg == "--batch_size") {
        params.nBatch = int.parse(argv[++i]);
      } else if (arg == "-m" || arg == "--model") {
        params.model = argv[++i];
      } else if (arg == "-i" || arg == "--interactive") {
        params.interactive = true;
      } else if (arg == "--interactive-start") {
        params.interactive = true;
        params.interactiveStart = true;
      } else if (arg == "--color") {
        params.useColor = true;
      } else if (arg == "-r" || arg == "--reverse-prompt") {
        params.antiPrompt = argv[++i];
      } else if (arg == "-h" || arg == "--help") {
        gptPrintUsage(argc, argv, params);
        exit(0);
      } else {
        print('error: unknown argument: $arg\n');
        gptPrintUsage(argc, argv, params);
        exit(0);
      }
    }

    return true;
  }

  static void gptPrintUsage(
      int argc, List<String> argv, AlpacaGptParams params) {
    print('usage: ${argv[0]} [options]\n');
    print('\n');
    print('options:\n');
    print('  -h, --help            show this help message and exit\n');
    print('  -i, --interactive     run in interactive mode\n');
    print(
        '  --interactive-start   run in interactive mode and poll user input at startup\n');
    print('  -r PROMPT, --reverse-prompt PROMPT\n');
    print(
        '                        in interactive mode, poll user input upon seeing PROMPT\n');
    print(
        '  --color               colorize output to distinguish prompt and user input from generations\n');
    print('  -s SEED, --seed SEED  RNG seed (default: -1)\n');
    print(
        '  -t N, --threads N     number of threads to use during computation (default: ${params.nThreads}\n');
    print('  -p PROMPT, --prompt PROMPT\n');
    print(
        '                        prompt to start generation with (default: random)\n');
    print('  -f FNAME, --file FNAME\n');
    print('                        prompt file to start generation.\n');
    print(
        '  -n N, --n_predict N   number of tokens to predict (default: ${params.nPredict})\n');
    print('  --top_k N             top-k sampling (default: ${params.topK})\n');
    print('  --top_p N             top-p sampling (default: ${params.topP})\n');
    print(
        '  --repeat_last_n N     last n tokens to consider for penalize (default: ${params.repeatLastN})\n');
    print(
        '  --repeat_penalty N    penalize repeat sequence of tokens (default: ${params.repeatPenalty})\n');
    print(
        '  -c N, --ctx_size N    size of the prompt context (default: ${params.nCtx})\n');
    print('  --temp N              temperature (default: ${params.temp})\n');
    print(
        '  -b N, --batch_size N  batch size for prompt processing (default: ${params.nBatch})\n');
    print('  -m FNAME, --model FNAME\n');
    print('                        model path (default: ${params.model})\n');
    print('\n');
  }

  /// SentencePiece implementation after https://guillaume-be.github.io/2020-05-30/sentence_piece
  static List<Id?> llamaTokenize(AlpacaGptVocab vocab, String text, bool bos) {
    List<Id?> res = [];
    final int len = text.length;
    final score = List<int>.filled(len + 1, 0);
    final prev = List<Id>.filled(len + 1, 0);
    // Forward pass
    for (int i = 0; i < len; i++) {
      for (int subLen = 1; subLen <= len - i; subLen++) {
        var sub = text.substring(i, i + subLen);
        var token = vocab.tokenToId.containsKey(sub);
        if (token) {
          int tokenScore = sub.length * sub.length;
          int localScore = score[i] + tokenScore;
          int next = i + subLen;
          if (score[next] < localScore) {
            score[next] = localScore;
            prev[next] = vocab.tokenToId[sub]!;
          }
        }
      }
    }

    // Backward pass
    int i = len;
    while (i > 0) {
      var tokenId = prev[i];
      if (tokenId == 0) {
        print("failed to tokenize string!\n");
        break;
      }
      res.add(tokenId);
      if (vocab.idToToken.containsKey(tokenId)) {
        var token = vocab.idToToken[tokenId];
        i -= token!.length;
      }
    }

    if (bos) {
      res.add(1);
    }

    // Pieces are in reverse order so correct that
    res = res.reversed.toList();

    return res;
  }

  static void sampleTopK(List<AlpacaGptLogit> logitsId, int topK) {
    if (topK == 0 || topK > logitsId.length) {
      return;
    }

    // Sort the list, highest first
    final toSort = logitsId.toList();
    toSort.sort();

    // Get the first topK elements
    final subLogits = toSort.sublist(0, topK);
    if (subLogits.isEmpty) {
      return;
    }

    // Remove the sorted sublist from logitsId
    final tmpList = logitsId.toList();
    tmpList.removeWhere((element) => subLogits.contains(element));

    // Add the temp list to the sorted list
    subLogits.addAll(tmpList);

    // Assign back to the input parameter
    logitsId
      ..clear()
      ..addAll(subLogits);
  }

  /// Sample next token given probabilities for each embedding
  ///
  ///   - consider only the top K tokens
  ///   - from them, consider only the top tokens with cumulative probability > P
  ///
  static Id llamaSampleTopPTopK(
      AlpacaGptVocab vocab,
      AlpacaLogit logits,
      List<Id> lastNTokens,
      double repeatPenalty,
      int topK,
      double topP,
      double temp,
      Random rng) {
    int nLogits = vocab.idToToken.length;

    final logitsId = <AlpacaGptLogit>[];
    {
      final double scale = 1.0 / temp;
      for (int i = 0; i < nLogits; ++i) {
        // Repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main

        if (lastNTokens.contains(i)) {
          // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability

          if (logits.logits[i] < 0.0) {
            logitsId.add(AlpacaGptLogit()
              ..id = i
              ..val = logits.logits[i] * scale * repeatPenalty);
          } else {
            logitsId.add(AlpacaGptLogit()
              ..id = i
              ..val = logits.logits[i] / scale * repeatPenalty);
          }
        } else {
          logitsId.add(AlpacaGptLogit()
            ..id = i
            ..val = logits.logits[i] / scale);
        }
      }
    }

    sampleTopK(logitsId, topK);

    double maxl = double.negativeInfinity;
    for (final entry in logitsId) {
      maxl = max(maxl, entry.val);
    }
    //
    // Compute probabilities for the top K tokens
    final probs = <double>[];
    //
    for (final entry in logitsId) {
      final p = exp(entry.val - maxl);
      probs.add(p);
    }
    //

    //
    if (topP < 1.0) {
      double cumsum = 0.0;
      for (int i = 0; i < probs.length; i++) {
        cumsum += probs[i];
      }

      cumsum = 1.0 / cumsum;
      for (int i = 0; i < probs.length; i++) {
        probs[i] *= cumsum;
      }
    }
    final dist =
        UniformDiscreteDistribution(probs.first.toInt(), probs.last.toInt());
    final idx = dist.sample(random: rng);
    return logitsId[idx].id;
  }
}
