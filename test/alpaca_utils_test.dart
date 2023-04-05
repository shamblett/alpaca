/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

@TestOn('vm && linux')

import 'package:test/test.dart';
import 'package:text_analysis/text_analysis.dart';
import 'package:alpaca/alpaca.dart';

int main() {
  test('Usage', () {
    AlpacaUtils.gptPrintUsage(1, ['chat'], AlpacaGptParams());
  });

  test('JSON parse', () {
    final decoded = AlpacaUtils.jsonParse('test/support/json_parse.json');
    expect(decoded['First'], 1);
    expect(decoded['Second'], 2);
  });

  test('Get llama tokenize', () async {
    final str = 'This is a string of words';
  });

  return 0;
}
