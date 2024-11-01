(component
  (component (;0;)
    (type (;0;)
      (instance
        (type (;0;) (tuple string string))
        (type (;1;) (list 0))
        (type (;2;) (func (result 1)))
        (export (;0;) "get-environment" (func (type 2)))
        (type (;3;) (list string))
        (type (;4;) (func (result 3)))
        (export (;1;) "get-arguments" (func (type 4)))
      )
    )
    (import "wasi:cli-base/environment" (instance (;0;) (type 0)))
    (type (;1;)
      (instance)
    )
    (import "other1" (instance (;1;) (type 1)))
    (type (;2;)
      (instance)
    )
    (import "other2" (instance (;2;) (type 2)))
    (core module (;0;)
      (type (;0;) (func (param i32)))
      (type (;1;) (func (param i32 i32 i32 i32) (result i32)))
      (import "wasi:cli-base/environment" "get-environment" (func (;0;) (type 0)))
      (import "wasi:cli-base/environment" "get-arguments" (func (;1;) (type 0)))
      (memory (;0;) 0)
      (export "memory" (memory 0))
      (export "cabi_realloc" (func 2))
      (func (;2;) (type 1) (param i32 i32 i32 i32) (result i32)
        unreachable
      )
      (@producers
        (processed-by "wit-component" "0.11.0")
      )
    )
    (core module (;1;)
      (type (;0;) (func (param i32)))
      (table (;0;) 2 2 funcref)
      (export "0" (func $indirect-wasi:cli-base/environment-get-environment))
      (export "1" (func $indirect-wasi:cli-base/environment-get-arguments))
      (export "$imports" (table 0))
      (func $indirect-wasi:cli-base/environment-get-environment (;0;) (type 0) (param i32)
        local.get 0
        i32.const 0
        call_indirect (type 0)
      )
      (func $indirect-wasi:cli-base/environment-get-arguments (;1;) (type 0) (param i32)
        local.get 0
        i32.const 1
        call_indirect (type 0)
      )
      (@producers
        (processed-by "wit-component" "0.11.0")
      )
    )
    (core module (;2;)
      (type (;0;) (func (param i32)))
      (import "" "0" (func (;0;) (type 0)))
      (import "" "1" (func (;1;) (type 0)))
      (import "" "$imports" (table (;0;) 2 2 funcref))
      (elem (;0;) (i32.const 0) func 0 1)
      (@producers
        (processed-by "wit-component" "0.11.0")
      )
    )
    (core instance (;0;) (instantiate 1))
    (alias core export 0 "0" (core func (;0;)))
    (alias core export 0 "1" (core func (;1;)))
    (core instance (;1;)
      (export "get-environment" (func 0))
      (export "get-arguments" (func 1))
    )
    (core instance (;2;) (instantiate 0
        (with "wasi:cli-base/environment" (instance 1))
      )
    )
    (alias core export 2 "memory" (core memory (;0;)))
    (alias core export 2 "cabi_realloc" (core func (;2;)))
    (alias core export 0 "$imports" (core table (;0;)))
    (alias export 0 "get-environment" (func (;0;)))
    (core func (;3;) (canon lower (func 0) (memory 0) (realloc 2) string-encoding=utf8))
    (alias export 0 "get-arguments" (func (;1;)))
    (core func (;4;) (canon lower (func 1) (memory 0) (realloc 2) string-encoding=utf8))
    (@producers
      (processed-by "wit-component" "0.11.0")
    )
    (core instance (;3;)
      (export "$imports" (table 0))
      (export "0" (func 3))
      (export "1" (func 4))
    )
    (core instance (;4;) (instantiate 2
        (with "" (instance 3))
      )
    )
  )
  (component (;1;)
    (core module (;0;)
      (type (;0;) (func))
      (type (;1;) (func (result i32)))
      (type (;2;) (func (param i32)))
      (type (;3;) (func (param i32 i32) (result i32)))
      (type (;4;) (func (param i32 i32 i32)))
      (type (;5;) (func (param i32 i32 i32 i32) (result i32)))
      (type (;6;) (func (param i32) (result i32)))
      (type (;7;) (func (param i32 i32)))
      (type (;8;) (func (param i32 i32 i32) (result i32)))
      (table (;0;) 1 1 funcref)
      (memory (;0;) 17)
      (global $__stack_pointer (;0;) (mut i32) i32.const 1048576)
      (export "memory" (memory 0))
      (export "wasi:cli-base/environment#get-environment" (func $wasi:cli-base/environment#get-environment))
      (export "cabi_post_wasi:cli-base/environment#get-environment" (func $cabi_post_wasi:cli-base/environment#get-environment))
      (export "wasi:cli-base/environment#get-arguments" (func $wasi:cli-base/environment#get-arguments))
      (export "cabi_post_wasi:cli-base/environment#get-arguments" (func $cabi_post_wasi:cli-base/environment#get-arguments))
      (export "cabi_realloc" (func $cabi_realloc))
      (func $__wasm_call_ctors (;0;) (type 0))
      (func $wasi:cli-base/environment#get-environment (;1;) (type 1) (result i32)
        i32.const 0
      )
      (func $cabi_post_wasi:cli-base/environment#get-environment (;2;) (type 2) (param i32))
      (func $wasi:cli-base/environment#get-arguments (;3;) (type 1) (result i32)
        i32.const 0
      )
      (func $cabi_post_wasi:cli-base/environment#get-arguments (;4;) (type 2) (param i32))
      (func $cabi_realloc (;5;) (type 5) (param i32 i32 i32 i32) (result i32)
        i32.const 0
      )
    )
    (core instance (;0;) (instantiate 0))
    (alias core export 0 "memory" (core memory (;0;)))
    (alias core export 0 "cabi_realloc" (core func (;0;)))
    (type (;0;) (tuple string string))
    (type (;1;) (list 0))
    (type (;2;) (func (result 1)))
    (alias core export 0 "wasi:cli-base/environment#get-environment" (core func (;1;)))
    (alias core export 0 "cabi_post_wasi:cli-base/environment#get-environment" (core func (;2;)))
    (func (;0;) (type 2) (canon lift (core func 1) (memory 0) string-encoding=utf8 (post-return 2)))
    (type (;3;) (list string))
    (type (;4;) (func (result 3)))
    (alias core export 0 "wasi:cli-base/environment#get-arguments" (core func (;3;)))
    (alias core export 0 "cabi_post_wasi:cli-base/environment#get-arguments" (core func (;4;)))
    (func (;1;) (type 4) (canon lift (core func 3) (memory 0) string-encoding=utf8 (post-return 4)))
    (component (;0;)
      (type (;0;) (tuple string string))
      (type (;1;) (list 0))
      (type (;2;) (func (result 1)))
      (import "import-func-get-environment" (func (;0;) (type 2)))
      (type (;3;) (list string))
      (type (;4;) (func (result 3)))
      (import "import-func-get-arguments" (func (;1;) (type 4)))
      (type (;5;) (tuple string string))
      (type (;6;) (list 5))
      (type (;7;) (func (result 6)))
      (export (;2;) "get-environment" (func 0) (func (type 7)))
      (type (;8;) (list string))
      (type (;9;) (func (result 8)))
      (export (;3;) "get-arguments" (func 1) (func (type 9)))
    )
    (instance (;0;) (instantiate 0
        (with "import-func-get-environment" (func 0))
        (with "import-func-get-arguments" (func 1))
      )
    )
    (export (;1;) "wasi:cli-base/environment" (instance 0))
  )
  (component (;2;)
    (instance (;0;))
    (export (;1;) "other1" (instance 0))
    (export (;2;) "other2" (instance 0))
  )
  (instance (;0;) (instantiate 2))
  (instance (;1;) (instantiate 1))
  (alias export 1 "wasi:cli-base/environment" (instance (;2;)))
  (alias export 0 "other1" (instance (;3;)))
  (alias export 0 "other2" (instance (;4;)))
  (instance (;5;) (instantiate 0
      (with "wasi:cli-base/environment" (instance 2))
      (with "other1" (instance 3))
      (with "other2" (instance 4))
    )
  )
)
