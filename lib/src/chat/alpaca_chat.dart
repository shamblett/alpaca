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

    // Load Hparams;
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

    try {
      raf.closeSync();
    } on FileSystemException {
      print('Failed to close file $fname');
      return false;
    }

    return true;
  }
}
