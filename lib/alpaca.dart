/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 02/04/2023
 * Copyright :  S.Hamblett
 */

library alpaca;

import 'dart:convert';
import 'dart:math';
import 'dart:io';
import 'dart:ffi';

import 'package:data/data.dart';
import 'package:ffi/ffi.dart' as ffi;

import 'src/ggml/implementation/ggml_impl.dart' as ggmlimpl;

part 'src/utils/api/alpaca_utils.dart';
part 'src/ggml/api/alpaca_ggml.dart';
part 'src/utils/types/alpaca_gpt_params.dart';
part 'src/utils/types/alpaca_gpt_vocab.dart';
part 'src/utils/types/alpaca_gpt_logit.dart';
part 'src/ggml/types/alpaca_ggml_object.dart';