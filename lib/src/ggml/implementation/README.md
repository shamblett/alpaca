# ggml_impl.so

There is currently a number(approx 6)hand edits that must be applied to the generated ggml_impl.dart file, example 

```
class ggml_init_params extends ffi.Struct {
// SJH - hand edit, ffigen is incorrectly setting this size_t
// member to ffi.Int(). This is too small for the memory size.
@ffi.Uint64()
external int mem_size;

external ffi.Pointer<ffi.Void> mem_buffer;
}
```

The mem_size member must be annotated as Uint64, so must any other type that is size_t in the ggml.h header.

# libggml.so

The ggml shared library is supplied in the library folder, however you will almost certainly need to build this for 
your system. this can be done in 2 ways :-

1. Clone the repository https://github.com/shamblett/alpaca.cpp.git, this is my forked repository of the alpaca.cpp repository.
   Simply type 'make', this will create the libggml.so library, copy this to the library folder in this directory.


2. Clone the https://github.com/antimatter15/alpaca.cpp repository, this will give you the latest ggml code, copy into 
   it the Makefile from the library directory and type 'make', copy the created libggml.so file to the library folder in this directory.

My cloned repository will always match the Dart release, the antimatter repository may not do however, you may
have to also copy the ggml.h header file into the ggml folder and regen the ggml_impl.dart file.

Ggml has been implemented as a standalone library and should be easily lifted and used elsewhere, in future this may become its own package.