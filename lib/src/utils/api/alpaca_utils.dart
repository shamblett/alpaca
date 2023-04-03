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
  /// Default uses the supplied dynamic library
  AlpacaUtils();

  /// API methods, lack of comment reflects lack in the C header file.

  // bool gptParamsParse(int argc, List<String> argv, AlpacaGptParams arg2) {
  //   final ret = _impl.gpt_params_parse(
  //     argc,
  //     _strListToCharPointer(argv),
  //     arg2,
  //   );
  //
  //   return ret != 0;
  // }

  // void gptPrintUsage(int argc, List<String> argv, AlpacaGptParams arg2) {
  //   _impl.gpt_print_usage(
  //     argc,
  //     _strListToCharPointer(argv),
  //     arg2,
  //   );
  // }

  String gptRandomPrompt() {
    int r = Random().nextInt(32767) % 10;
    switch (r) {
      case 0:
        return "So";
      case 1:
        return "Once upon a time";
      case 2:
        return "When";
      case 3:
        return "The";
      case 4:
        return "After";
      case 5:
        return "If";
      case 6:
        return "import";
      case 7:
        return "He";
      case 8:
        return "She";
      case 9:
        return "They";
      default:
        return "To";
    }
  }
}
