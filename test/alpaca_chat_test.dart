/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

@TestOn('vm && linux')

import 'dart:io';

import 'package:test/test.dart';
import 'package:alpaca/alpaca.dart';

int main() {
  test('Read model', () {
    AlpacaGptVocab? vocab;
    AlpacaLlamaModel? model;
    final fname = 'test/support/model/ggml-model-q4_0.bin';
    final ret = AlpacaChat.llamaModelLoad(fname, model, vocab, 1);
    expect(ret, isTrue);
  });

  return 0;
}
