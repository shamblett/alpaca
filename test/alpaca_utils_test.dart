/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

@TestOn('vm && linux')

import 'package:test/test.dart';

import 'package:alpaca/alpaca.dart';

int main() {
  test('Usage', () {
    final utils = AlpacaUtils();
    utils.gptPrintUsage(1, ['chat'], AlpacaGptParams());
  });
  test('Random prompt', () {
    final utils = AlpacaUtils();
    for (int i = 0; i < 10; i++) {
      print(utils.gptRandomPrompt());
    }
  });
  return 0;
}
