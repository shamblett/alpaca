/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 02/04/2023
 * Copyright :  S.Hamblett
 */

library ggml;

import 'dart:ffi';

import 'package:ffi/ffi.dart' as ffi;

import 'implementation/ggml_impl.dart' as ggmlimpl;

part 'api/ggml.dart';
part 'types/ggml_opaque.dart';
part 'types/ggml_type.dart';
part 'types/ggml_init_params.dart';
part 'types/ggml_tensor.dart';
part 'types/ggml_scratch.dart';
