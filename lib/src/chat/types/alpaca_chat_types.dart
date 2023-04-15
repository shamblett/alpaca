/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of alpaca;

const String ansiColorRed = 'x1b[31m';
const String ansiColorGreen = 'x1b[32m';
const String ansiColorYellow = 'x1b[33m';
const String ansiColorBlue = 'x1b[34m';
const String ansiColorMagenta = 'x1b[35m';
const String ansiColorCyan = 'x1b[36m';
const String ansiColorReset = 'x1b[0m';
const String ansiBold = 'x1b[1m';

// Determine number of model parts based on the dimension
const llamaNParts = <int, int>{4096: 1, 5120: 1, 6656: 1, 8192: 1};

// Default hparams (LLaMA 7B)
class AlpacaLlamaHParams {
  int nVocab = 32000;
  int nCtx = 512; // this is provided as user input?
  int nEmbd = 4096;
  int nMult = 256;
  int nHead = 32;
  int nLayer = 32;
  int nRot = 64;
  int f16 = 1;

  @override
  toString() =>
      'nVocab = $nVocab, nCtx = $nCtx, nEmbd = $nEmbd, nMult = $nMult, nHead = $nHead, nLayer = $nLayer, nRot = $nRot, f16 = $f16 \n';
}

class AlpacaLlamaLayer {
  // Normalization
  GgmlTensor? attentionNorm;

  // Attention
  GgmlTensor? wq;
  GgmlTensor? wk;
  GgmlTensor? wv;
  GgmlTensor? wo;

  // Normalization
  GgmlTensor? ffnNorm;

  // ff
  GgmlTensor? w1;
  GgmlTensor? w2;
  GgmlTensor? w3;
}

class AlpacaLlamaModel {
  AlpacaLlamaHParams? hParams = AlpacaLlamaHParams();

  GgmlTensor? tokEmbeddings;

  GgmlTensor? norm;
  GgmlTensor? output;

  final layers = <AlpacaLlamaLayer>[];

  // Key + value memory
  GgmlTensor? memoryK;
  GgmlTensor? memoryV;

  GgmlContext? ctx;
  final tensors = <String, GgmlTensor>{};
}
