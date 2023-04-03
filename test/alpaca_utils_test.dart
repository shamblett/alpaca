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
  test('Initialise', () {
    final utils = AlpacaUtils();
  });
  return 0;
}
