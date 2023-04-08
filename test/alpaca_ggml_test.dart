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
  test('Initialize', () {
    final ggml = AlpacaGgml();
  });

  test('Print Object', () {
    final ggml = AlpacaGgml();
    final obj = AlpacaGgmlObject();
    ggml.ggmlPrintObject(obj);
  });

  return 0;
}
