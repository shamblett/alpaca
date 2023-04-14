/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of alpaca;

class AlpacaChat {
  /// Load the model's weights from a file
  static bool llamaModelLoad(
      String fname, AlpacaLlamaModel? model, AlpacaGptVocab? vocab, int nCtx) {
    print('Loading model from $fname - please wait ...\n');

    File file = File(fname);
    RandomAccessFile raf;
    int bPos = 0;
    try {
      raf = file.openSync(mode: FileMode.read);
      raf.setPositionSync(bPos);
    } on FileSystemException {
      print('Failed to open $fname - exiting');
      return false;
    }

    final buff = raf.readSync(1024 * 1024);

    // verify magic
    final bData = ByteData.sublistView(buff);
    final magic = bData.getUint32(0, Endian.little);
    if (magic != 0x67676d6c) {
      print('Invalid model file - bad magic $magic');
      return false;
    }
    bPos += 4;

    int nFf = 0;
    int nParts = 0;

    // Load hParams;
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

    // Load vocab
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
    print('Vocab size is ${vocab!.idToToken.length}');

    // For the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation.
    try {
      raf.closeSync();
    } on FileSystemException {
      print('Failed to close file $fname');
      return false;
    }

    return true;
  }
}
