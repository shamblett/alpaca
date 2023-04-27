/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

@TestOn('vm && linux')

import 'dart:ffi';
import 'package:test/test.dart';
import 'package:ffi/ffi.dart' as ffi;

import 'package:alpaca/src/ggml/ggml.dart';

int main() {
  test('Initialize', () {
    final ggml = Ggml();
    final initParams = GgmlInitParams()..instance;
    final ctx = ggml.init(initParams);
    expect(ctx, isNotNull);
    ggml.free(ctx);
    initParams.free();
  });

  group('Tensor', () {
    const bufSize = 512 * 1024 * 1024;
    final bufPtr = ffi.calloc<Uint8>(bufSize);
    final params = GgmlInitParams();
    params.instance.mem_size = bufSize;
    params.instance.mem_buffer = bufPtr.cast<Void>();

    test('1D Data type I32', () {
      final ggml = Ggml();
      final ctx = ggml.init(params);
      final tensor = ggml.newTensor1D(ctx, GgmlType.i32, 4);
      ggml.setI321d(tensor, 0, 1);
      final dPtr = ggml.getData(tensor).cast<Int>();
      expect(dPtr[0], 1);
      ggml.free(ctx);
    });

    test('1D Data type I32 - multi', () {
      final ggml = Ggml();
      final ctx = ggml.init(params);
      final tensor = ggml.newTensor1D(ctx, GgmlType.i32, 4);
      for (int i = 0; i < 5; i++) {
        ggml.setI321d(tensor, i, i);
      }
      final values = tensor.getTopXDataInt();
      expect(values, [0, 1, 2, 3, 4]);
      ggml.free(ctx);
    });

    test('1D Data type F32', () {
      final ggml = Ggml();
      final ctx = ggml.init(params);
      final tensor = ggml.newTensor1D(ctx, GgmlType.f32, 4);
      ggml.setF321d(tensor, 0, 1.0);
      final dPtr = ggml.getDataF32(tensor);
      expect(dPtr[0], 1.0);
      ggml.free(ctx);
    });

    test('1D Data type F32 - multi', () {
      final ggml = Ggml();
      final ctx = ggml.init(params);
      final tensor = ggml.newTensor1D(ctx, GgmlType.f32, 4);
      for (int i = 0; i < 5; i++) {
        ggml.setF321d(tensor, i, i.toDouble());
      }
      final values = tensor.getTopXDataDouble();
      expect(values, [0.0, 1.0, 2.0, 3.0, 4.0]);
      ggml.free(ctx);
    });

    test('1D Data type I32 - set data', () {
      final ggml = Ggml();
      final ctx = ggml.init(params);
      final tensor = ggml.newTensor1D(ctx, GgmlType.i32, 4);
      final vals = [0, 1, 2, 3, 4];
      tensor.setDataInt(vals);
      final values = tensor.getTopXDataInt();
      expect(values, vals);
      ggml.free(ctx);
    });

    test('1D Data type F32 - set data', () {
      final ggml = Ggml();
      final ctx = ggml.init(params);
      final vals = [0.0, 1.0, 2.0, 3.0, 4.0];
      final tensor = ggml.newTensor1D(ctx, GgmlType.f32, 4);
      tensor.setDataDouble(vals);
      final values = tensor.getTopXDataDouble();
      expect(values, vals);
      ggml.free(ctx);
    });
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
