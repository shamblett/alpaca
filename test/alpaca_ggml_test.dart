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
      print(tensor.dump());
      final dPtr = ggml.getData(tensor).cast<Int>();
      print('Value = ${dPtr[0]}');
    });

    test('1D Data type I32 - multi', () {
      final ggml = Ggml();
      final ctx = ggml.init(params);
      final tensor = ggml.newTensor1D(ctx, GgmlType.i32, 4);
      for (int i = 0; i < 5; i++) {
        ggml.setI321d(tensor, i, i);
      }
      print(tensor.dump());
      final dPtr = ggml.getData(tensor).cast<Int>();
      for (int i = 0; i < 5; i++) {
        print('Value = ${dPtr[i]}');
      }
    });

    test('1D Data type F32', () {
      final ggml = Ggml();
      final ctx = ggml.init(params);
      final tensor = ggml.newTensor1D(ctx, GgmlType.f32, 4);
      ggml.setF321d(tensor, 0, 1.0);
      print(tensor.dump());
      final dPtr = ggml.getDataF32(tensor);
      print('Value = ${dPtr[0]}');
    });

    test('1D Data type F32 - multi', () {
      final ggml = Ggml();
      final ctx = ggml.init(params);
      final tensor = ggml.newTensor1D(ctx, GgmlType.f32, 4);
      for (int i = 0; i < 5; i++) {
        ggml.setF321d(tensor, i, i.toDouble());
      }
      print(tensor.dump());
      final dPtr = ggml.getDataF32(tensor);
      for (int i = 0; i < 5; i++) {
        print('Value = ${dPtr[i]}');
      }
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
