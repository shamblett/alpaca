/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

@TestOn('vm && linux')

import 'package:test/test.dart';

import 'package:alpaca/alpaca.dart';
import 'package:alpaca/src/ggml/ggml.dart';

int main() {
  test('Read model', () {
    final vocab = AlpacaGptVocab();
    final model = AlpacaLlamaModel();
    final gptParams = AlpacaGptParams();
    final fname = 'test/support/model/ggml-model-q4_0.bin';
    final ggml = Ggml();
    final ret = AlpacaChat.llamaModelLoad(fname, model, vocab, gptParams.nCtx, ggml);
    expect(ret, isTrue);
    print(model.hParams);
  });

  return 0;
}
