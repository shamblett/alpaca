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

  test('Random prompt', () {
    for (int i = 0; i < 10; i++) {
      print(AlpacaUtils.gptRandomPrompt());
    }
  });

  test('Replace', () {
    var str = 'was 23';
    final pass = <String>[str];
    AlpacaUtils.replace(pass, '23', '24');
    expect(pass[0], 'was 24');
  });

  test('JSON parse', () {
    final decoded = AlpacaUtils.jsonParse('test/support/json_parse.json');
    expect(decoded['First'], 1);
    expect(decoded['Second'], 2);
  });

  test('Get tokenize', () {
    final str = 'This is a string of words';
    final words = AlpacaUtils.gptTokenize(AlpacaGptVocab(), str);
    expect(words, ['This', 'is', 'a', 'string', 'of', 'words']);
  });

  return 0;
}
