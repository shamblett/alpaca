/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of alpaca;

class AlpacaChat {
  /// Load the model's weights from a file
  static bool llamaModelLoad(String fname, AlpacaLlamaModel? model,
      AlpacaGptVocab? vocab, int nCtx, Ggml ggml) {
    print('Loading model from $fname - please wait ...');
    const latin1Decoder = Latin1Decoder();

    File file = File(fname);
    RandomAccessFile raf;
    int bPos = 0; // Starts at 0 and rolls on down the method
    Uint8List buff;
    try {
      raf = file.openSync(mode: FileMode.read);
      raf.setPositionSync(bPos);
      buff = raf.readSync(1024 * 1024);
    } on FileSystemException {
      print('llamaModelLoad:: Failed to open "$fname" - exiting');
      return false;
    }

    // Verify magic
    final bData = ByteData.sublistView(buff);
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

    const GgmlType wType2 = GgmlType.f32;
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

    // Create the ggml context
    {
      final params = GgmlInitParams();
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
      final nCtx = hParams.nCtx;
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
    try {
      raf.closeSync();
    } on FileSystemException {
      print(
          'llamaModelLoad:: Failed to close file for part processing "$fname"');
      return false;
    }

    for (int i = 0; i < nParts!; ++i) {
      final partId = i;
      var fnamePart = fname;
      if (i > 0) {
        fnamePart += '.$i';
      }

      print(
          'llamaModelLoad:: loading model part ${i + 1}/$nParts from "$fnamePart"');
      int fileLength = 0;
      Uint8List partBuff1;
      Uint8List partBuff2;
      // Load the rest of the model file. Split it into two chunks
      // and concatenate them. Dart doesn't seem to want to read Random Access files
      // in chunks greater that 2GB.
      try {
        // Read the buffers
        raf = file.openSync(mode: FileMode.read);
        fileLength = raf.lengthSync();
        raf.setPositionSync(0);
        final lengthToRead = fileLength;
        partBuff1 = raf.readSync(lengthToRead ~/ 2);
        raf.setPositionSync(lengthToRead ~/ 2);
        int oddOffset = fileLength.isOdd ? 1 : 0;
        partBuff2 = raf.readSync((lengthToRead ~/ 2) + oddOffset);
      } on FileSystemException {
        print('llamaModelLoad:: Failed to open "$fname" - exiting');
        return false;
      }

      // Concatenate the buffers
      final partBuff = Uint8List(partBuff1.length + partBuff2.length);
      partBuff.setAll(0, partBuff1);
      partBuff.setAll(partBuff1.length, partBuff2);
      final bData = ByteData.view(partBuff.buffer);

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
              tensor.setData(bytes);
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
              final np0 = ne[0];

              final rowSize = (tensor.instance.ne[0] /
                  ggml.blockSize(GgmlType.type(tensor.instance.type)) *
                  ggml.typeSize(GgmlType.type(tensor.instance.type)));
              assert(rowSize == tensor.instance.nb[1]);

              for (int i1 = 0; i1 < ne[1]; ++i1) {
                final offsetRow = i1 * rowSize;
                final offset = offsetRow +
                    ((partId * np0) /
                            ggml.blockSize(
                                GgmlType.type(tensor.instance.type))) *
                        ggml.typeSize(GgmlType.type(tensor.instance.type));
                //TODO needs implementing for other models - fin.read(reinterpret_cast<char *>(tensor->data) + offset, row_size/n_parts);
              }
            } else {
              final np1 = ne[1];

              final rowSize = (tensor.instance.ne[0] /
                      ggml.blockSize(GgmlType.type(tensor.instance.type))) *
                  ggml.typeSize(GgmlType.type(tensor.instance.type));

              for (int i1 = 0; i1 < ne[1]; ++i1) {
                final offsetRow = (i1 + partId * np1) * rowSize;
                // TODO needs implementing for other models - fin.read(reinterpret_cast<char *>(tensor->data) + offset_row, row_size);
              }
            }

            totalSize += ggml.nBytes(tensor) ~/ nParts;
          }

          if (++nTensors % 8 == 0) {
            stdout.write('.');
          }
        }
        stdout.flush();
        print('');
        print('llamaModelLoad:: done');

        print(
            'llamaModelLoad:: model size = ${totalSize / 1024.0 ~/ 1024.0} MB / num tensors = $nTensors');
      }
    }

    try {
      raf.closeSync();
    } on FileSystemException {
      print('llamaModelLoad:: Failed to close file $fname');
      return false;
    }

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
      List<Id> embdInp, List<double> embdW, int memPerToken) {
    final N = embdInp.length;

    final hParams = model.hParams;

    final nEmbd = hParams?.nEmbd;
    final nLayer = hParams?.nLayer;
    final nCtx = hParams?.nCtx;
    final nHead = hParams?.nHead;
    final nVocab = hParams?.nVocab;
    final nRot = nEmbd! / nHead!;

    final dKey = nEmbd! / nHead!;

    // TODO: check if this size scales with n_ctx linearly and remove constant. somehow I feel it wasn't the case
    // static size_t buf_size = hparams.n_ctx*1024*1024;
    const bufSize = 512 * 1024 * 1024;
    final bufPtr = ffi.calloc<Uint8>(bufSize);

    final params = GgmlInitParams();
    params.instance.mem_size = bufSize;
    params.instance.mem_buffer = bufPtr.cast<Void>();

    final ggml = Ggml();
    final ctx0 = ggml.init(params);
    final gf = GgmlCGraph();
    gf.instance.n_threads = nThreads;

    final embd = ggml.newTensor1D(ctx0, GgmlType.i32, N);
    embd.setData(Uint8List.fromList(embdInp));

    final inpL = ggml.getRows(ctx0, model.tokEmbeddings, embd);

    return false;
  }
}
