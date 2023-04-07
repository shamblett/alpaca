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
  void logitMatcher(List<AlpacaGptLogit> values, List<double> match) {
    for (int x = 0; x < values.length; x++) {
      expect(values[x].val, match[x]);
    }
  }

  test('Usage', () {
    AlpacaUtils.gptPrintUsage(1, ['chat'], AlpacaGptParams());
  });

  test('Get Llama tokenize', () async {
    final str = 'This is a string of words';
  });

  test('Sample Top K 0', () {
    final values = <AlpacaGptLogit>[
      AlpacaGptLogit()
        ..id = 1
        ..val = 5.0,
      AlpacaGptLogit()
        ..id = 2
        ..val = 7.0,
      AlpacaGptLogit()
        ..id = 3
        ..val = 4.0,
      AlpacaGptLogit()
        ..id = 4
        ..val = 2.0,
      AlpacaGptLogit()
        ..id = 5
        ..val = 8.0,
      AlpacaGptLogit()
        ..id = 6
        ..val = 6.0,
      AlpacaGptLogit()
        ..id = 7
        ..val = 1.0,
      AlpacaGptLogit()
        ..id = 8
        ..val = 9.0,
      AlpacaGptLogit()
        ..id = 9
        ..val = 0.0,
      AlpacaGptLogit()
        ..id = 10
        ..val = 3.0,
    ];

    AlpacaUtils.sampleTopK(values, 0);
    logitMatcher(values, [5.0, 7.0, 4.0, 2.0, 8.0, 6.0, 1.0, 9.0, 0.0, 3.0]);
  });

  test('Sample Top K 3', () {
    final values = <AlpacaGptLogit>[
      AlpacaGptLogit()
        ..id = 1
        ..val = 5.0,
      AlpacaGptLogit()
        ..id = 2
        ..val = 7.0,
      AlpacaGptLogit()
        ..id = 3
        ..val = 4.0,
      AlpacaGptLogit()
        ..id = 4
        ..val = 2.0,
      AlpacaGptLogit()
        ..id = 5
        ..val = 8.0,
      AlpacaGptLogit()
        ..id = 6
        ..val = 6.0,
      AlpacaGptLogit()
        ..id = 7
        ..val = 1.0,
      AlpacaGptLogit()
        ..id = 8
        ..val = 9.0,
      AlpacaGptLogit()
        ..id = 9
        ..val = 0.0,
      AlpacaGptLogit()
        ..id = 10
        ..val = 3.0,
    ];

    AlpacaUtils.sampleTopK(values, 3);
    logitMatcher(values, [9.0, 8.0, 7.0, 5.0, 4.0, 2.0, 6.0, 1.0, 0.0, 3.0]);
  });

  test('Sample Top K 7', () {
    final values = <AlpacaGptLogit>[
      AlpacaGptLogit()
        ..id = 1
        ..val = 5.0,
      AlpacaGptLogit()
        ..id = 2
        ..val = 7.0,
      AlpacaGptLogit()
        ..id = 3
        ..val = 4.0,
      AlpacaGptLogit()
        ..id = 4
        ..val = 2.0,
      AlpacaGptLogit()
        ..id = 5
        ..val = 8.0,
      AlpacaGptLogit()
        ..id = 6
        ..val = 6.0,
      AlpacaGptLogit()
        ..id = 7
        ..val = 1.0,
      AlpacaGptLogit()
        ..id = 8
        ..val = 9.0,
      AlpacaGptLogit()
        ..id = 9
        ..val = 0.0,
      AlpacaGptLogit()
        ..id = 10
        ..val = 3.0,
    ];

    AlpacaUtils.sampleTopK(values, 7);
    logitMatcher(values, [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
  });

  test('Sample Top K 15', () {
    final values = <AlpacaGptLogit>[
      AlpacaGptLogit()
        ..id = 1
        ..val = 5.0,
      AlpacaGptLogit()
        ..id = 2
        ..val = 7.0,
      AlpacaGptLogit()
        ..id = 3
        ..val = 4.0,
      AlpacaGptLogit()
        ..id = 4
        ..val = 2.0,
      AlpacaGptLogit()
        ..id = 5
        ..val = 8.0,
      AlpacaGptLogit()
        ..id = 6
        ..val = 6.0,
      AlpacaGptLogit()
        ..id = 7
        ..val = 1.0,
      AlpacaGptLogit()
        ..id = 8
        ..val = 9.0,
      AlpacaGptLogit()
        ..id = 9
        ..val = 0.0,
      AlpacaGptLogit()
        ..id = 10
        ..val = 3.0,
    ];

    AlpacaUtils.sampleTopK(values, 15);
    logitMatcher(values, [5.0, 7.0, 4.0, 2.0, 8.0, 6.0, 1.0, 9.0, 0.0, 3.0]);
  });

  return 0;
}
