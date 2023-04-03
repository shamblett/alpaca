/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

part of alpaca;

class AlpacaGptParams {
  int seed = -1; // RNG seed
  int nThreads = 1;
  int nPredict = 128; // new tokens to predict
  int repeatLastN = 64; // last n tokens to penalize
  int nCtx = 2048; //context size

  // sampling parameters
  int topK = 40;
  double topP = 0.95;
  double temp = 0.10;
  double repeatPenalty = 1.30;

  int nBatch = 8; // batch size for prompt processing

  String model = "ggml-alpaca-7b-q4.bin"; // model path
  String prompt = '';

  bool useColor = true; // use color to distinguish generations and inputs

  bool interactive = true; // interactive mode
  bool interactiveStart = true; // reverse prompt immediately
  String antiPrompt =
      ''; // string upon seeing which more user input is prompted
}
