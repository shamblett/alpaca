/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 04/04/2023
 * Copyright :  S.Hamblett
 */

part of ggml;

class GgmlInitParams {
  ggmlimpl.ggml_init_params get instance {
    final initParamsPtr = ffi.calloc<ggmlimpl.ggml_init_params>();
    final initParams = initParamsPtr.ref;
    return initParams;
  }
}
