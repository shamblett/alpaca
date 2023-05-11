/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of alpaca;

class AlpacaChat {
  static final ggml = Ggml();
  static var gf = GgmlCGraph();
  static int memPerToken = 0;

  // Don't recreate these variables everytime eval is called, also
  // expose them for clearing after an eval pass if needed.

  static late GgmlTensor embd;

  /// Load the model's weights from a file
  static bool llamaModelLoad(String fname, AlpacaLlamaModel? model,
      AlpacaGptVocab? vocab, int nCtx, Ggml ggml) {
    print('Loading model from $fname - please wait ...');
    const latin1Decoder = Latin1Decoder();

    // Native file ops
    final bufFd = stdlib.open(fname);
    if (bufFd < 0) {
      print('llamaModelLoad:: cannot open model file [$fname]');
      return false;
    }
    final buff = stdlib.read(bufFd, 1024 * 1024);
    if (buff.isEmpty) {
      print('llamaModelLoad:: failed to read from model file [$fname]');
      return false;
    }
    int bPos = 0; // Starts at 0 and rolls on down the method

    // Verify magic
    final bData = ByteData.sublistView(Uint8List.fromList(buff));
    final magic = bData.getUint32(0, Endian.little);
    if (magic != 0x67676d6c) {
      print('llamaModelLoad:: Invalid model file - bad magic $magic');
      return false;
    }
    bPos += 4;

    int nFf = 0;
    int? nParts = 0;

    // Load hParams;
    {
      model!.hParams!.nVocab = bData.getInt32(bPos, Endian.little);
      bPos += 4;
      model.hParams!.nEmbd = bData.getInt32(bPos, Endian.little);
      bPos += 4;
      model.hParams!.nMult = bData.getInt32(bPos, Endian.little);
      bPos += 4;
      model.hParams!.nHead = bData.getInt32(bPos, Endian.little);
      bPos += 4;
      model.hParams!.nLayer = bData.getInt32(bPos, Endian.little);
      bPos += 4;
      model.hParams!.nRot = bData.getInt32(bPos, Endian.little);
      bPos += 4;
      model.hParams!.f16 = bData.getInt32(bPos, Endian.little);
      bPos += 4;
      model.hParams!.nCtx = nCtx;

      final hParams = model.hParams!;

      final t1 =
          (2 * (4 * hParams.nEmbd) / 3 + hParams.nMult - 1) ~/ hParams.nMult;
      nFf = t1 * hParams.nMult;
      nParts = llamaNParts[hParams.nEmbd];
    }

    // Load vocab
    {
      final nVocab = model.hParams!.nVocab;
      for (int i = 0; i < nVocab; i++) {
        int len = bData.getInt32(bPos, Endian.little);
        bPos += 4;
        final chars = bData.buffer.asUint8List(bPos, len);
        final word = latin1Decoder.convert(chars);
        vocab!.tokenToId[word] = i;
        vocab.idToToken[i] = word;
        bPos += len;
      }
    }

    // For the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation.
    var wType = GgmlType.count;
    switch (model.hParams!.f16) {
      case 0:
        wType = GgmlType.f32;
        break;
      case 1:
        wType = GgmlType.f16;
        break;
      case 2:
        wType = GgmlType.q40;
        break;
      case 3:
        wType = GgmlType.q41;
        break;
      default:
        {
          print(
              'llamaModelLoad:: Invalid model file "$fname" (bad f16 value ${model.hParams!.f16})');
          return false;
        }
    }

    int ctxSizeInt = 0;
    {
      final hParams = model.hParams;
      final nEmbd = hParams!.nEmbd;
      final nLayer = hParams.nLayer;
      final nCtx = hParams.nCtx;
      final nVocab = hParams.nVocab;
      double ctxSize = 0.0;

      ctxSize += nEmbd * nVocab * ggml.typeSizeF(wType); // tok_embeddings
      ctxSize += nEmbd * ggml.typeSizeF(GgmlType.f32); // norm
      ctxSize += nEmbd * nVocab * ggml.typeSizeF(wType); // output
      ctxSize +=
          nLayer * (nEmbd * ggml.typeSizeF(GgmlType.f32)); // attention_norm
      //
      ctxSize += nLayer * (nEmbd * nEmbd * ggml.typeSizeF(wType)); // wq
      ctxSize += nLayer * (nEmbd * nEmbd * ggml.typeSizeF(wType)); // wk
      ctxSize += nLayer * (nEmbd * nEmbd * ggml.typeSizeF(wType)); // wv
      ctxSize += nLayer * (nEmbd * nEmbd * ggml.typeSizeF(wType)); // wo
      //
      ctxSize +=
          nLayer * (nEmbd * ggml.typeSizeF(GgmlType.f32)).toInt(); // ffn_norm
      //
      ctxSize += nLayer * (nFf * nEmbd * ggml.typeSizeF(wType)); // w1
      ctxSize += nLayer * (nFf * nEmbd * ggml.typeSizeF(wType)); // w2
      ctxSize += nLayer * (nFf * nEmbd * ggml.typeSizeF(wType)); // w3
      //
      ctxSize +=
          nCtx * nLayer * nEmbd * ggml.typeSizeF(GgmlType.f32); // memory_k
      ctxSize +=
          nCtx * nLayer * nEmbd * ggml.typeSizeF(GgmlType.f32); // memory_v
      //
      ctxSize += (5 + 10 * nLayer) * 256; // object overhead
      //
      print(
          'llamaModelLoad:: Ggml ctx size = ${ctxSize ~/ (1024.0 * 1024.0)} MB');
      ctxSizeInt = ctxSize.toInt();
    }

    final params = GgmlInitParams();

    // Create the ggml context
    {
      params.instance.mem_size = ctxSizeInt;
      params.instance.mem_buffer = nullptr;
      model.ctx = nullptr;
      model.ctx = ggml.init(params);
      if (model.ctx == nullptr) {
        print('llamaModelLoad:: ggml.init() failed');
        return false;
      }
    }

    // Prepare memory for the weights
    {
      final hParams = model.hParams;
      final nEmbd = hParams!.nEmbd;
      final nLayer = hParams.nLayer;
      final nVocab = hParams.nVocab;
      final ctx = model.ctx;

      model.tokEmbeddings = ggml.newTensor2D(ctx!, wType, nEmbd, nVocab);
      model.norm = ggml.newTensor1D(ctx, GgmlType.f32, nEmbd);
      model.output = ggml.newTensor2D(ctx, wType, nEmbd, nVocab);

      // Map by name
      model.tensors["tok_embeddings.weight"] = model.tokEmbeddings!;
      model.tensors["norm.weight"] = model.norm!;
      model.tensors["output.weight"] = model.output!;

      for (int i = 0; i < nLayer; ++i) {
        final layer = AlpacaLlamaLayer();

        layer.attentionNorm = ggml.newTensor1D(ctx, GgmlType.f32, nEmbd);

        layer.wq = ggml.newTensor2D(ctx, wType, nEmbd, nEmbd);
        layer.wk = ggml.newTensor2D(ctx, wType, nEmbd, nEmbd);
        layer.wv = ggml.newTensor2D(ctx, wType, nEmbd, nEmbd);
        layer.wo = ggml.newTensor2D(ctx, wType, nEmbd, nEmbd);

        layer.ffnNorm = ggml.newTensor1D(ctx, GgmlType.f32, nEmbd);

        layer.w1 = ggml.newTensor2D(ctx, wType, nEmbd, nFf);
        layer.w2 = ggml.newTensor2D(ctx, wType, nFf, nEmbd);
        layer.w3 = ggml.newTensor2D(ctx, wType, nEmbd, nFf);

        // Map by name
        model.tensors['layers.$i.attention_norm.weight'] = layer.attentionNorm!;
        model.tensors['layers.$i.attention.wq.weight'] = layer.wq!;
        model.tensors['layers.$i.attention.wk.weight'] = layer.wk!;
        model.tensors['layers.$i.attention.wv.weight'] = layer.wv!;
        model.tensors['layers.$i.attention.wo.weight'] = layer.wo!;
        model.tensors['layers.$i.ffn_norm.weight'] = layer.ffnNorm!;
        model.tensors['layers.$i.feed_forward.w1.weight'] = layer.w1!;
        model.tensors['layers.$i.feed_forward.w2.weight'] = layer.w2!;
        model.tensors['layers.$i.feed_forward.w3.weight'] = layer.w3!;

        model.layers.add(layer);
      }
    }

    // Key + value memory
    {
      final hParams = model.hParams;

      final nEmbd = hParams!.nEmbd;
      final nLayer = hParams.nLayer;
      final nCtx = hParams.nCtx;
      final ctx = model.ctx;

      final nMem = nLayer * nCtx;
      final nElements = nEmbd * nMem;

      model.memoryK = ggml.newTensor1D(ctx!, GgmlType.f32, nElements);
      model.memoryV = ggml.newTensor1D(ctx, GgmlType.f32, nElements);

      final memorySize =
          ggml.nBytes(model.memoryK!) + ggml.nBytes(model.memoryV!);

      print(
          'llamaModelLoad:: Memory_size = ${memorySize / 1024.0 / 1024.0} MB, nMem = $nMem');
    }

    // Load model parts;
    for (int i = 0; i < nParts!; ++i) {
      final partId = i;
      var fnamePart = fname;
      if (i > 0) {
        fnamePart += '.$i';
      }

      print(
          'llamaModelLoad:: loading model part ${i + 1}/$nParts from "$fnamePart"');
      int fileLength = stdlib.stat(fnamePart)!.st_size;
      if (fileLength <= 0) {
        print(
            'llamaModelLoad:: failed to stat model part ${i + 1}/$nParts from "$fnamePart"');
        return false;
      }
      final partFd = stdlib.open(fnamePart);
      if (partFd < 0) {
        print(
            'llamaModelLoad:: failed to open model part ${i + 1}/$nParts from "$fnamePart"');
        return false;
      }
      final pBufMapped = stdlib.mmap(
          length: fileLength,
          fd: partFd,
          prot: stdlib.PROT_READ,
          flags: stdlib.MAP_PRIVATE);
      if (pBufMapped!.data.lengthInBytes <= 0) {
        print(
            'llamaModelLoad:: failed to mmap model part ${i + 1}/$nParts from "$fnamePart"');
      }
      final bData = ByteData.view(pBufMapped.data);

      // Load weights
      {
        int nTensors = 0;
        int totalSize = 0;
        while (bPos < fileLength) {
          int nDims = -1;
          int length = -1;
          int fType = -1;
          try {
            nDims = bData.getInt32(bPos, Endian.little);
            bPos += 4;
          } catch (e) {
            print(e);
          }
          try {
            length = bData.getInt32(bPos, Endian.little);
            bPos += 4;
          } catch (e) {
            print(e);
          }
          try {
            fType = bData.getInt32(bPos, Endian.little);
            bPos += 4;
          } catch (e) {
            print(e);
          }

          int nElements = 1;
          final ne = <int>[1, 1];
          for (int i = 0; i < nDims; ++i) {
            ne[i] = bData.getInt32(bPos, Endian.little);
            bPos += 4;
            nElements *= ne[i];
          }

          final chars = bData.buffer.asUint8List(bPos, length);
          final name = latin1Decoder.convert(chars);
          bPos += length;

          if (!model.tensors.containsKey(name)) {
            print('llamaModelLoad:: unknown tensor "$name" in model file');
            return false;
          }

          int splitType = 0;
          if (name.contains('tok_embeddings')) {
            splitType = 0;
          } else if (name.contains('layers')) {
            if (name.contains('attention.wo.weight')) {
              splitType = 0;
            } else if (name.contains('feed_forward.w2.weight')) {
              splitType = 0;
            } else {
              splitType = 1;
            }
          } else if (name.contains('output')) {
            splitType = 1;
          }

          final tensor = model.tensors[name];

          if (nDims == 1) {
            if (ggml.nElements(tensor!) != nElements) {
              print(
                  'llamaModelLoad:: 1 tensor "$name" has wrong size in model file');
              return false;
            }
          } else {
            final te = ggml.nElements(tensor!);
            if (te / nParts != nElements) {
              print(
                  'llamaModelLoad:: 2 tensor "$name" has wrong size in model file');
              return false;
            }
          }

          if (nDims == 1) {
            if (tensor.instance.ne[0] != ne[0] ||
                tensor.instance.ne[1] != ne[1]) {
              print(
                  'llamaModelLoad:: tensor "$name" has wrong shape in model file: got [${tensor.instance.ne[0]},${tensor.instance.ne[1]}], expected [${ne[0]}, ${ne[1]}]');
              return false;
            }
          } else {
            if (splitType == 0) {
              if (tensor.instance.ne[0] / nParts != ne[0] ||
                  tensor.instance.ne[1] != ne[1]) {
                print(
                    'llamaModelLoad:: tensor "$name" has wrong shape in model file: got [${tensor.instance.ne[0] / nParts},${tensor.instance.ne[1]}], expected [${ne[0]}, ${ne[1]}]');
                return false;
              }
            } else {
              if (tensor.instance.ne[0] != ne[0] ||
                  tensor.instance.ne[1] / nParts != ne[1]) {
                print(
                    'llamaModelLoad:: tensor "$name" has wrong shape in model file: got [${tensor.instance.ne[0]},${tensor.instance.ne[1] / nParts}], expected [${ne[0]}, ${ne[1]}]');
                return false;
              }
            }
          }

          int bpe = 0;

          switch (fType) {
            case 0:
              bpe = ggml.typeSize(GgmlType.f32);
              break;
            case 1:
              bpe = ggml.typeSize(GgmlType.f16);
              break;
            case 2:
              bpe = ggml.typeSize(GgmlType.q40);
              assert(ne[0] % 64 == 0);
              break;
            case 3:
              bpe = ggml.typeSize(GgmlType.q41);
              assert(ne[0] % 64 == 0);
              break;
            default:
              {
                print('llamaModelLoad:: Unknown ftype [$fType] in model file');
                return false;
              }
          }

          if (nDims == 1 || nParts == 1) {
            if ((nElements * bpe) /
                    ggml.blockSize(GgmlType.type(tensor.instance.type)) !=
                ggml.nBytes(tensor)) {
              print(
                  'llamaModelLoad:: 1 tensor "$name" has wrong size in model file: got ${ggml.nBytes(tensor)}, expected ${nElements * bpe}');
              return false;
            }
            final readLength = ggml.nBytes(tensor);
            if (partId == 0) {
              final bytes = bData.buffer.asUint8List(bPos, readLength);
              tensor.setDataBytes(bytes);
            }
            bPos += readLength;
            totalSize += ggml.nBytes(tensor);
          } else {
            if ((nElements * bpe) /
                    ggml.blockSize(GgmlType.type(tensor.instance.type)) !=
                ggml.nBytes(tensor) / nParts) {
              print(
                  'llamaModelLoad:: 2 tensor "$name" has wrong size in model file: got ${ggml.nBytes(tensor) / nParts}, expected ${nElements * bpe}');
              return false;
            }

            if (splitType == 0) {
              final rowSize = (tensor.instance.ne[0] /
                  ggml.blockSize(GgmlType.type(tensor.instance.type)) *
                  ggml.typeSize(GgmlType.type(tensor.instance.type)));
              assert(rowSize == tensor.instance.nb[1]);
            }

            totalSize += ggml.nBytes(tensor) ~/ nParts;
          }

          if (++nTensors % 8 == 0) {
            stdout.write('.');
          }
        }
        print('');
        print('llamaModelLoad:: done');

        print(
            'llamaModelLoad:: model size = ${totalSize / 1024.0 ~/ 1024.0} MB / num tensors = $nTensors');
      }
      stdlib.close(partFd);
      stdlib.munmap(pBufMapped);
    }
    stdlib.close(bufFd);

    return true;
  }

  static String llamaPrintSystemInfo(Ggml ggml) {
    final s = StringBuffer();

    s.write('');
    s.write('AVX = ${ggml.cpuHasAvx()}  |  ');
    s.write('AVX2 = ${ggml.cpuHasAvx2()}  |  ');
    s.write('AVX512 = ${ggml.cpuHasAvx512()}  |  ');
    s.write('FMA = ${ggml.cpuHasFma()} | ');
    s.write('NEON = ${ggml.cpuHasNeon()} | ');
    s.write('ARM_FMA = ${ggml.cpuHasArmFma()} | ');
    s.write('F16C = ${ggml.cpuHasF16c()} | ');
    s.write('FP16_VA = ${ggml.cpuHasFp16Va()} | ');
    s.write('WASM_SIMD = ${ggml.cpuHasWasmSimd()} | ');
    s.write('BLAS = ${ggml.cpuHasBlas()} | ');
    s.write('SSE3 = ${ggml.cpuHasSse3()} | ');
    s.write('VSX = ${ggml.cpuHasVsx()} | ');

    return s.toString();
  }

  /// Evaluate the transformer
  ///
  ///  - model:     the model
  ///  - nThreads: number of threads to use
  ///  - nPast:    the context size so far
  ///  - embdInp:  the embeddings of the tokens in the context
  ///  - embdW:    the predicted logits for the next token
  ///
  /// The GPT-J model requires about 16MB of memory per input token.
  static bool llamaEval(AlpacaLlamaModel model, int nThreads, int nPast,
      List<Id> embdInp, AlpacaLogit embdW) {
    final N = embdInp.length;

    final hParams = model.hParams;

    final nEmbd = hParams?.nEmbd;
    final nLayer = hParams?.nLayer;
    final nCtx = hParams?.nCtx;
    final nHead = hParams?.nHead;
    final nVocab = hParams?.nVocab;
    final nRot = nEmbd! / nHead!;

    const bufSize = 512 * 1024 * 1024;
    final bufPtr = ffi.calloc<Uint8>(bufSize);

    final params = GgmlInitParams();
    params.instance.mem_size = bufSize;
    params.instance.mem_buffer = bufPtr.cast<Void>();

    final ctx0 = ggml.init(params);
    gf = GgmlCGraph();
    gf.instance.n_threads = nThreads;

    embd = ggml.newTensor1D(ctx0, GgmlType.i32, N);
    embd.setDataInt(embdInp);

    var inpL = ggml.getRows(ctx0, model.tokEmbeddings!, embd);

    for (int il = 0; il < nLayer!; ++il) {
      final inpSA = inpL;
      var cur = GgmlTensor();
      // Norm
      {
        cur = ggml.rmsNorm(ctx0, inpL);

        // cur = attention_norm*cur
        cur = ggml.mul(
            ctx0, ggml.repeat(ctx0, model.layers[il].attentionNorm!, cur), cur);
      }

      // Self-attention
      {
        final qCur = ggml.mulMat(ctx0, model.layers[il].wq!, cur);
        final kCur = ggml.mulMat(ctx0, model.layers[il].wk!, cur);
        final vCur = ggml.mulMat(ctx0, model.layers[il].wv!, cur);

        // store key and value to memory
        if (N >= 1) {
          final k = ggml.view1D(
              ctx0,
              model.memoryK!,
              N * nEmbd,
              (ggml.elementSize(model.memoryK!) * nEmbd) *
                  (il * nCtx! + nPast));
          final v = ggml.view1D(ctx0, model.memoryV!, N * nEmbd,
              (ggml.elementSize(model.memoryV!) * nEmbd) * (il * nCtx + nPast));
          ggml.buildForwardExpand(gf, ggml.cpy(ctx0, kCur, k));
          ggml.buildForwardExpand(gf, ggml.cpy(ctx0, vCur, v));
        }

        // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
        final q = ggml.permute(
            ctx0,
            ggml.rope(
                ctx0,
                ggml.cpy(
                    ctx0,
                    qCur,
                    ggml.newTensor3D(
                        ctx0, GgmlType.f32, nEmbd ~/ nHead, nHead, N)),
                nPast,
                nRot.toInt(),
                0),
            0,
            2,
            1,
            3);

        // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
        final k = ggml.permute(
            ctx0,
            ggml.rope(
                ctx0,
                ggml.reshape3D(
                    ctx0,
                    ggml.view1D(ctx0, model.memoryK!, (nPast + N) * nEmbd,
                        il * nCtx! * ggml.elementSize(model.memoryK!) * nEmbd),
                    nEmbd ~/ nHead,
                    nHead,
                    nPast + N),
                nPast,
                nRot.toInt(),
                1),
            0,
            2,
            1,
            3);

        // K * Q
        final kq = ggml.mulMat(ctx0, k, q);

        // KQ_scaled = KQ / sqrt(n_embd/n_head)
        final yy = 1.0 / sqrt(nEmbd / nHead);
        final kqScaled = ggml.scale(ctx0, kq, ggml.newF32(ctx0, yy));

        // KQ_masked = mask_past(KQ_scaled)
        final kqMasked = ggml.diagMaskInf(ctx0, kqScaled, nPast);

        // KQ = soft_max(KQ_masked)
        final kqSoftMax = ggml.softMax(ctx0, kqMasked);

        // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
        final vTrans = ggml.permute(
            ctx0,
            ggml.reshape3D(
                ctx0,
                ggml.view1D(ctx0, model.memoryV!, (nPast + N) * nEmbd,
                    il * nCtx * ggml.elementSize(model.memoryV!) * nEmbd),
                nEmbd ~/ nHead,
                nHead,
                nPast + N),
            1,
            2,
            0,
            3);

        // KQV = transpose(V) * KQ_soft_max
        final kqv = ggml.mulMat(ctx0, vTrans, kqSoftMax);

        // KQV_merged = KQV.permute(0, 2, 1, 3)
        final kqvMerged = ggml.permute(ctx0, kqv, 0, 2, 1, 3);

        cur = ggml.cpy(
            ctx0, kqvMerged, ggml.newTensor2D(ctx0, GgmlType.f32, nEmbd, N));

        // Projection (no bias)
        cur = ggml.mulMat(ctx0, model.layers[il].wo!, cur);
      }

      final inpFF = ggml.add(ctx0, cur, inpSA);

      // Feed-forward network
      {
        // Norm
        {
          cur = ggml.rmsNorm(ctx0, inpFF);

          // cur = ffn_norm*cur
          cur = ggml.mul(
              ctx0, ggml.repeat(ctx0, model.layers[il].ffnNorm!, cur), cur);
        }

        final tmp = ggml.mulMat(ctx0, model.layers[il].w3!, cur);

        cur = ggml.mulMat(ctx0, model.layers[il].w1!, cur);

        // SILU activation
        cur = ggml.silu(ctx0, cur);

        cur = ggml.mul(ctx0, cur, tmp);

        cur = ggml.mulMat(ctx0, model.layers[il].w2!, cur);
      }

      cur = ggml.add(ctx0, cur, inpFF);

      // Input for next layer
      inpL = cur;
    }

    // Norm
    {
      inpL = ggml.rmsNorm(ctx0, inpL);

      // inpL = norm*inpL
      inpL = ggml.mul(ctx0, ggml.repeat(ctx0, model.norm!, inpL), inpL);
    }

    // lm_head
    {
      inpL = ggml.mulMat(ctx0, model.output!, inpL);
    }

    // Run the computation
    ggml.buildForwardExpand(gf, inpL);
    ggml.graphCompute(ctx0, gf);

    // Return result for just the last token;
    final resPtr = inpL.getDataF32().elementAt(nVocab! * (N - 1));
    AlpacaLogit.logits = resPtr.asTypedList(nVocab);

    if (memPerToken == 0) {
      memPerToken = ggml.usedMem(ctx0) ~/ N;
    }

    gf.free();
    ggml.free(ctx0);

    return true;
  }
}
