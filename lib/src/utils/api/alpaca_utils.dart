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

  static String gptRandomPrompt() {
    int r = Random().nextInt(32767) % 10;
    switch (r) {
      case 0:
        return 'So';
      case 1:
        return 'Once upon a time';
      case 2:
        return 'When';
      case 3:
        return 'The';
      case 4:
        return 'After';
      case 5:
        return 'If';
      case 6:
        return 'import';
      case 7:
        return 'He';
      case 8:
        return 'She';
      case 9:
        return 'They';
      default:
        return 'To';
    }
  }

  static void replace(List<String> str, String needle, String replacement) {
    str[0] = str[0].replaceAll(needle, replacement);
  }

  static Map<String, int> jsonParse(String fileName) {
    final jsonFile = File(fileName);
    final jsonString = jsonFile.readAsStringSync();
    return JsonDecoder().convert(jsonString).cast<String, int>();
  }

  ///
  /// Split text into tokens
  ///
  /// ref: https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L53
  ///
  /// Regex (Python):
  /// r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  ///
  /// Regex (C++):
  /// R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)"
  ///
  static List<String?> gptTokenize(AlpacaGptVocab vocab, String text) {
    // First split the text into words
    final words = <String?>[];
    var str = text;
    final pat =
        r"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

    final re = RegExp(pat);
    final m = re.allMatches(str).toList();
    for (var i = 0; i < m.length; i++) {
      for (var k = 0; k < m[i].groupCount; k++) {
        words.add(m[i].group(k));
      }
    }

    // find the longest tokens that form the words:
    // std::vector<gpt_vocab::id> tokens;
    // for (const auto & word : words) {
    // if (word.size() == 0) continue;
    //
    // int i = 0;
    // int n = word.size();
    // while (i < n) {
    // int j = n;
    // while (j > i) {
    // auto it = vocab.token_to_id.find(word.substr(i, j-i));
    // if (it != vocab.token_to_id.end()) {
    // tokens.push_back(it->second);
    // i = j;
    // break;
    // }
    // --j;
    // }
    // if (i == n) {
    // break;
    // }
    // if (j == i) {
    // auto sub = word.substr(i, 1);
    // if (vocab.token_to_id.find(sub) != vocab.token_to_id.end()) {
    // tokens.push_back(vocab.token_to_id.at(sub));
    // } else {
    // fprintf(stderr, "%s: unknown token '%s'\n", __func__, sub.data());
    // }
    // ++i;
    // }
    // }
    // }

    return words;
  }
}
