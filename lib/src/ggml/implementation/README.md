# Warning

There is currently a hand edit that must be applied to the generated ggml_impl.dart file, 

```
class ggml_init_params extends ffi.Struct {
// SJH - hand edit, ffigen is incorrectly setting this size_t
// member to ffi.Int(). This is too small for the memory size.
@ffi.Uint64()
external int mem_size;

external ffi.Pointer<ffi.Void> mem_buffer;
}
```

The mem_size member must be annotated as Uint64