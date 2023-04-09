/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

@TestOn('vm && linux')

import 'package:test/test.dart';

import 'package:alpaca/src/ggml/ggml.dart';

int main() {
  test('Initialize', () {
    Ggml();
  });

  return 0;
}
