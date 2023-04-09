/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

typedef GgmlObject = Pointer<ggmlimpl.ggml_object>;
typedef GgmlContext = Pointer<ggmlimpl.ggml_context>;
typedef GgmlTensor = ggmlimpl.ggml_tensor;
