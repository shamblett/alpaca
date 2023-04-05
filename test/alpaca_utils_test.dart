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
    AlpacaUtils.gptPrintUsage(1, ['chat'], AlpacaGptParams());
  });

  test('Get llama tokenize', () async {
    final str = 'This is a string of words';
  });

  return 0;
}
