/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 02/04/2023
 * Copyright :  S.Hamblett
 */

library alpaca;

import 'dart:convert';
import 'dart:ffi';
import 'dart:math';
import 'dart:io';
import 'dart:typed_data';

import 'package:data/data.dart';

import 'package:alpaca/src/ggml/ggml.dart';

part 'src/utils/api/alpaca_utils.dart';
part 'src/utils/types/alpaca_gpt_params.dart';
part 'src/utils/types/alpaca_gpt_vocab.dart';
part 'src/utils/types/alpaca_gpt_logit.dart';
part 'src/chat/types/alpaca_chat_types.dart';
part 'src/chat/alpaca_chat.dart';
