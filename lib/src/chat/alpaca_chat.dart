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
    print('Loading model from $fname - please wait ...\n');

    File file = File(fname);
    RandomAccessFile raf;
    int bPos = 0;
    try {
      raf = file.openSync(mode: FileMode.read);
      raf.setPositionSync(bPos);
    } on FileSystemException {
      print('llamaModelLoad:: Failed to open $fname - exiting');
      return false;
    }

    final buff = raf.readSync(1024 * 1024);

    // verify magic
    final bData = ByteData.sublistView(buff);
    final magic = bData.getUint32(0, Endian.little);
    if (magic != 0x67676d6c) {
      print('llamaModelLoad:: Invalid model file - bad magic $magic');
      return false;
    }
    bPos += 4;

    int nFf = 0;
    int nParts = 0;

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
    }
    // Load vocab
    {
      const latin1Decoder = Latin1Decoder();
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
    print('llamaModelLoad:: Vocab size is ${vocab!.idToToken.length}');

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
              'llamaModelLoad:: Invalid model file $fname (bad f16 value ${model.hParams!.f16})\n');
          return false;
        }
    }

    const GgmlType wType2 = GgmlType.f32;
    int ctxSize = 0;
    {
      final hParams = model.hParams;
      final nEmbd = hParams!.nEmbd;
      final nLayer = hParams.nLayer;
      final nCtx = hParams.nCtx;
      final nVocab = hParams.nVocab;

      ctxSize +=
          nEmbd * nVocab * ggml.typeSizeF(wType).toInt(); // tok_embeddings
      ctxSize += nEmbd * ggml.typeSizeF(GgmlType.f32).toInt(); // norm
      ctxSize += nEmbd * nVocab * ggml.typeSizeF(wType).toInt(); // output
      ctxSize += nLayer *
          (nEmbd * ggml.typeSizeF(GgmlType.f32)).toInt(); // attention_norm
      //
      ctxSize += nLayer * (nEmbd * nEmbd * ggml.typeSizeF(wType)).toInt(); // wq
      ctxSize += nLayer * (nEmbd * nEmbd * ggml.typeSizeF(wType)).toInt(); // wk
      ctxSize += nLayer * (nEmbd * nEmbd * ggml.typeSizeF(wType)).toInt(); // wv
      ctxSize += nLayer * (nEmbd * nEmbd * ggml.typeSizeF(wType)).toInt(); // wo
      //
      ctxSize +=
          nLayer * (nEmbd * ggml.typeSizeF(GgmlType.f32)).toInt(); // ffn_norm
      //
      ctxSize += nLayer * (nFf * nEmbd * ggml.typeSizeF(wType)).toInt(); // w1
      ctxSize += nLayer * (nFf * nEmbd * ggml.typeSizeF(wType)).toInt(); // w2
      ctxSize += nLayer * (nFf * nEmbd * ggml.typeSizeF(wType)).toInt(); // w3
      //
      ctxSize += nCtx *
          nLayer *
          nEmbd *
          ggml.typeSizeF(GgmlType.f32).toInt(); // memory_k
      ctxSize += nCtx *
          nLayer *
          nEmbd *
          ggml.typeSizeF(GgmlType.f32).toInt(); // memory_v
      //
      ctxSize += (5 + 10 * nLayer) * 256; // object overhead
      //
      print(
          'llamaModelLoad:: Ggml ctx size = ${ctxSize ~/ (1024.0 * 1024.0)} MB\n');
    }

    // Create the ggml context
    {
      final params = GgmlInitParams();
      params.instance.mem_size = ctxSize;
      params.instance.mem_buffer = nullptr;
      model.ctx = nullptr;
      model.ctx = ggml.init(params);
      if (model.ctx == nullptr) {
        print('llamaModelLoad:: ggml.init() failed\n');
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

    try {
      raf.closeSync();
    } on FileSystemException {
      print('llamaModelLoad:: Failed to close file $fname');
      return false;
    }

    return true;
  }
}
