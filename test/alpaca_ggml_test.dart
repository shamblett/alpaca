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
    final ggml = Ggml();
    final initParams = GgmlInitParams()..instance;
    final ctx = ggml.init(initParams);
    expect(ctx, isNotNull);
    print(ggml.usedMem(ctx));
    //ggml.free(ctx);
    //initParams.free();
  });

  test('Various', () {
    final ggml = Ggml();
    var ret = ggml.blockSize(GgmlType.q40);
    expect(ret, 32);
    ret = ggml.typeSize(GgmlType.q40);
    expect(ret, 20);
    var retF = ggml.typeSizeF(GgmlType.q40);
    expect(retF, 0.625);
  });

  return 0;
}
