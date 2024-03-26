use crate::component::*;
use crate::{ExportKind, Module, RawSection, ValType};
use std::mem;

/// Convenience type to build a component incrementally and automatically keep
/// track of index spaces.
///
/// This type is intended to be a wrapper around the [`Component`] encoding type
/// which is useful for building it up incrementally over time. This type will
/// automatically collect definitions into sections and reports the index of all
/// items added by keeping track of indices internally.
#[derive(Debug, Default)]
pub struct ComponentBuilder {
    /// The binary component that's being built.
    component: Component,

    /// The last section which was appended to during encoding. This type is
    /// generated by the `section_accessors` macro below.
    ///
    /// When something is encoded this is used if it matches the kind of item
    /// being encoded, otherwise it's "flushed" to the output component and a
    /// new section is started.
    last_section: LastSection,

    // Core index spaces
    core_modules: u32,
    core_funcs: u32,
    core_types: u32,
    core_memories: u32,
    core_tables: u32,
    core_instances: u32,
    core_tags: u32,
    core_globals: u32,

    // Component index spaces
    funcs: u32,
    instances: u32,
    types: u32,
    components: u32,
    values: u32,
}

impl ComponentBuilder {
    /// Returns the current number of core modules.
    pub fn core_module_count(&self) -> u32 {
        self.core_modules
    }

    /// Returns the current number of core funcs.
    pub fn core_func_count(&self) -> u32 {
        self.core_funcs
    }

    /// Returns the current number of core types.
    pub fn core_type_count(&self) -> u32 {
        self.core_types
    }

    /// Returns the current number of core memories.
    pub fn core_memory_count(&self) -> u32 {
        self.core_memories
    }

    /// Returns the current number of core tables.
    pub fn core_table_count(&self) -> u32 {
        self.core_tables
    }

    /// Returns the current number of core instances.
    pub fn core_instance_count(&self) -> u32 {
        self.core_instances
    }

    /// Returns the current number of core tags.
    pub fn core_tag_count(&self) -> u32 {
        self.core_tags
    }

    /// Returns the current number of core globals.
    pub fn core_global_count(&self) -> u32 {
        self.core_globals
    }

    /// Returns the current number of component funcs.
    pub fn func_count(&self) -> u32 {
        self.funcs
    }

    /// Returns the current number of component instances.
    pub fn instance_count(&self) -> u32 {
        self.instances
    }

    /// Returns the current number of component values.
    pub fn value_count(&self) -> u32 {
        self.values
    }

    /// Returns the current number of components.
    pub fn component_count(&self) -> u32 {
        self.components
    }

    /// Returns the current number of component types.
    pub fn type_count(&self) -> u32 {
        self.types
    }

    /// Completes this component and returns the binary encoding of the entire
    /// component.
    pub fn finish(mut self) -> Vec<u8> {
        self.flush();
        self.component.finish()
    }

    /// Encodes a core wasm `Module` into this component, returning its index.
    pub fn core_module(&mut self, module: &Module) -> u32 {
        self.flush();
        self.component.section(&ModuleSection(module));
        inc(&mut self.core_modules)
    }

    /// Encodes a core wasm `module` into this component, returning its index.
    pub fn core_module_raw(&mut self, module: &[u8]) -> u32 {
        self.flush();
        self.component.section(&RawSection {
            id: ComponentSectionId::CoreModule.into(),
            data: module,
        });
        inc(&mut self.core_modules)
    }

    /// Instantiates a core wasm module at `module_index` with the `args`
    /// provided.
    ///
    /// Returns the index of the core wasm instance crated.
    pub fn core_instantiate<'a, A>(&mut self, module_index: u32, args: A) -> u32
    where
        A: IntoIterator<Item = (&'a str, ModuleArg)>,
        A::IntoIter: ExactSizeIterator,
    {
        self.instances().instantiate(module_index, args);
        inc(&mut self.core_instances)
    }

    /// Creates a new core wasm instance from the `exports` provided.
    ///
    /// Returns the index of the core wasm instance crated.
    pub fn core_instantiate_exports<'a, E>(&mut self, exports: E) -> u32
    where
        E: IntoIterator<Item = (&'a str, ExportKind, u32)>,
        E::IntoIter: ExactSizeIterator,
    {
        self.instances().export_items(exports);
        inc(&mut self.core_instances)
    }

    /// Creates a new aliased item where the core `instance` specified has its
    /// export `name` aliased out with the `kind` specified.
    ///
    /// Returns the index of the item crated.
    pub fn core_alias_export(&mut self, instance: u32, name: &str, kind: ExportKind) -> u32 {
        self.alias(Alias::CoreInstanceExport {
            instance,
            kind,
            name,
        })
    }

    /// Adds a new alias to this component
    pub fn alias(&mut self, alias: Alias<'_>) -> u32 {
        self.aliases().alias(alias);
        match alias {
            Alias::InstanceExport { kind, .. } => self.inc_kind(kind),
            Alias::CoreInstanceExport { kind, .. } => self.inc_core_kind(kind),
            Alias::Outer {
                kind: ComponentOuterAliasKind::Type,
                ..
            } => inc(&mut self.types),
            Alias::Outer {
                kind: ComponentOuterAliasKind::CoreModule,
                ..
            } => inc(&mut self.core_modules),
            Alias::Outer {
                kind: ComponentOuterAliasKind::Component,
                ..
            } => inc(&mut self.components),
            Alias::Outer {
                kind: ComponentOuterAliasKind::CoreType,
                ..
            } => inc(&mut self.core_types),
        }
    }

    /// Creates an alias to a previous component instance's exported item.
    ///
    /// The `instance` provided is the instance to access and the `name` is the
    /// item to access.
    ///
    /// Returns the index of the new item defined.
    pub fn alias_export(&mut self, instance: u32, name: &str, kind: ComponentExportKind) -> u32 {
        self.alias(Alias::InstanceExport {
            instance,
            kind,
            name,
        })
    }

    fn inc_kind(&mut self, kind: ComponentExportKind) -> u32 {
        match kind {
            ComponentExportKind::Func => inc(&mut self.funcs),
            ComponentExportKind::Module => inc(&mut self.core_modules),
            ComponentExportKind::Type => inc(&mut self.types),
            ComponentExportKind::Component => inc(&mut self.components),
            ComponentExportKind::Instance => inc(&mut self.instances),
            ComponentExportKind::Value => inc(&mut self.values),
        }
    }

    fn inc_core_kind(&mut self, kind: ExportKind) -> u32 {
        match kind {
            ExportKind::Func => inc(&mut self.core_funcs),
            ExportKind::Table => inc(&mut self.core_tables),
            ExportKind::Memory => inc(&mut self.core_memories),
            ExportKind::Global => inc(&mut self.core_globals),
            ExportKind::Tag => inc(&mut self.core_tags),
        }
    }

    /// Lowers the `func_index` component function into a core wasm function
    /// using the `options` provided.
    ///
    /// Returns the index of the core wasm function created.
    pub fn lower_func<O>(&mut self, func_index: u32, options: O) -> u32
    where
        O: IntoIterator<Item = CanonicalOption>,
        O::IntoIter: ExactSizeIterator,
    {
        self.canonical_functions().lower(func_index, options);
        inc(&mut self.core_funcs)
    }

    /// Lifts the core wasm `core_func_index` function with the component
    /// function type `type_index` and `options`.
    ///
    /// Returns the index of the component function created.
    pub fn lift_func<O>(&mut self, core_func_index: u32, type_index: u32, options: O) -> u32
    where
        O: IntoIterator<Item = CanonicalOption>,
        O::IntoIter: ExactSizeIterator,
    {
        self.canonical_functions()
            .lift(core_func_index, type_index, options);
        inc(&mut self.funcs)
    }

    /// Imports a new item into this component with the `name` and `ty` specified.
    pub fn import(&mut self, name: &str, ty: ComponentTypeRef) -> u32 {
        let ret = match &ty {
            ComponentTypeRef::Instance(_) => inc(&mut self.instances),
            ComponentTypeRef::Func(_) => inc(&mut self.funcs),
            ComponentTypeRef::Type(..) => inc(&mut self.types),
            ComponentTypeRef::Component(_) => inc(&mut self.components),
            ComponentTypeRef::Module(_) => inc(&mut self.core_modules),
            ComponentTypeRef::Value(_) => inc(&mut self.values),
        };
        self.imports().import(name, ty);
        ret
    }

    /// Exports a new item from this component with the `name` and `kind`
    /// specified.
    ///
    /// The `idx` is the item to export and the `ty` is an optional type to
    /// ascribe to the export.
    pub fn export(
        &mut self,
        name: &str,
        kind: ComponentExportKind,
        idx: u32,
        ty: Option<ComponentTypeRef>,
    ) -> u32 {
        self.exports().export(name, kind, idx, ty);
        self.inc_kind(kind)
    }

    /// Creates a new encoder for the next core type in this component.
    pub fn core_type(&mut self) -> (u32, ComponentCoreTypeEncoder<'_>) {
        (inc(&mut self.core_types), self.core_types().ty())
    }

    /// Creates a new encoder for the next type in this component.
    pub fn ty(&mut self) -> (u32, ComponentTypeEncoder<'_>) {
        (inc(&mut self.types), self.types().ty())
    }

    /// Creates a new instance type within this component.
    pub fn type_instance(&mut self, ty: &InstanceType) -> u32 {
        self.types().instance(ty);
        inc(&mut self.types)
    }

    /// Creates a new component type within this component.
    pub fn type_component(&mut self, ty: &ComponentType) -> u32 {
        self.types().component(ty);
        inc(&mut self.types)
    }

    /// Creates a new defined component type within this component.
    pub fn type_defined(&mut self) -> (u32, ComponentDefinedTypeEncoder<'_>) {
        (inc(&mut self.types), self.types().defined_type())
    }

    /// Creates a new component function type within this component.
    pub fn type_function(&mut self) -> (u32, ComponentFuncTypeEncoder<'_>) {
        (inc(&mut self.types), self.types().function())
    }

    /// Declares a new resource type within this component.
    pub fn type_resource(&mut self, rep: ValType, dtor: Option<u32>) -> u32 {
        self.types().resource(rep, dtor);
        inc(&mut self.types)
    }

    /// Defines a new subcomponent of this component.
    pub fn component(&mut self, mut builder: ComponentBuilder) -> u32 {
        builder.flush();
        self.flush();
        self.component
            .section(&NestedComponentSection(&builder.component));
        inc(&mut self.components)
    }

    /// Defines a new subcomponent of this component.
    pub fn component_raw(&mut self, data: &[u8]) -> u32 {
        let raw_section = RawSection {
            id: ComponentSectionId::Component.into(),
            data,
        };
        self.flush();
        self.component.section(&raw_section);
        inc(&mut self.components)
    }

    /// Instantiates the `component_index` specified with the `args` specified.
    pub fn instantiate<A, S>(&mut self, component_index: u32, args: A) -> u32
    where
        A: IntoIterator<Item = (S, ComponentExportKind, u32)>,
        A::IntoIter: ExactSizeIterator,
        S: AsRef<str>,
    {
        self.component_instances()
            .instantiate(component_index, args);
        inc(&mut self.instances)
    }

    /// Declares a new `resource.drop` intrinsic.
    pub fn resource_drop(&mut self, ty: u32) -> u32 {
        self.canonical_functions().resource_drop(ty);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `resource.new` intrinsic.
    pub fn resource_new(&mut self, ty: u32) -> u32 {
        self.canonical_functions().resource_new(ty);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `resource.rep` intrinsic.
    pub fn resource_rep(&mut self, ty: u32) -> u32 {
        self.canonical_functions().resource_rep(ty);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `thread.spawn` intrinsic.
    pub fn thread_spawn(&mut self, ty: u32) -> u32 {
        self.canonical_functions().thread_spawn(ty);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `thread.hw_concurrency` intrinsic.
    pub fn thread_hw_concurrency(&mut self) -> u32 {
        self.canonical_functions().thread_hw_concurrency();
        inc(&mut self.core_funcs)
    }

    /// Declares a new `task.backpressure` intrinsic.
    pub fn task_backpressure(&mut self) -> u32 {
        self.canonical_functions().task_backpressure();
        inc(&mut self.core_funcs)
    }

    /// Declares a new `task.return` intrinsic.
    pub fn task_return(&mut self, ty: u32) -> u32 {
        self.canonical_functions().task_return(ty);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `task.wait` intrinsic.
    pub fn task_wait(&mut self, async_: bool, memory: u32) -> u32 {
        self.canonical_functions().task_wait(async_, memory);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `task.poll` intrinsic.
    pub fn task_poll(&mut self, async_: bool, memory: u32) -> u32 {
        self.canonical_functions().task_poll(async_, memory);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `task.yield` intrinsic.
    pub fn task_yield(&mut self, async_: bool) -> u32 {
        self.canonical_functions().task_yield(async_);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `subtask.drop` intrinsic.
    pub fn subtask_drop(&mut self) -> u32 {
        self.canonical_functions().subtask_drop();
        inc(&mut self.core_funcs)
    }

    /// Declares a new `stream.new` intrinsic.
    pub fn stream_new(&mut self, ty: u32) -> u32 {
        self.canonical_functions().stream_new(ty);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `stream.read` intrinsic.
    pub fn stream_read<O>(&mut self, ty: u32, options: O) -> u32
    where
        O: IntoIterator<Item = CanonicalOption>,
        O::IntoIter: ExactSizeIterator,
    {
        self.canonical_functions().stream_read(ty, options);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `stream.write` intrinsic.
    pub fn stream_write<O>(&mut self, ty: u32, options: O) -> u32
    where
        O: IntoIterator<Item = CanonicalOption>,
        O::IntoIter: ExactSizeIterator,
    {
        self.canonical_functions().stream_write(ty, options);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `stream.cancel-read` intrinsic.
    pub fn stream_cancel_read(&mut self, ty: u32, async_: bool) -> u32 {
        self.canonical_functions().stream_cancel_read(ty, async_);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `stream.cancel-write` intrinsic.
    pub fn stream_cancel_write(&mut self, ty: u32, async_: bool) -> u32 {
        self.canonical_functions().stream_cancel_read(ty, async_);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `stream.close-readable` intrinsic.
    pub fn stream_close_readable(&mut self, ty: u32) -> u32 {
        self.canonical_functions().stream_close_readable(ty);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `stream.close-writable` intrinsic.
    pub fn stream_close_writable(&mut self, ty: u32) -> u32 {
        self.canonical_functions().stream_close_writable(ty);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `future.new` intrinsic.
    pub fn future_new(&mut self, ty: u32) -> u32 {
        self.canonical_functions().future_new(ty);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `future.read` intrinsic.
    pub fn future_read<O>(&mut self, ty: u32, options: O) -> u32
    where
        O: IntoIterator<Item = CanonicalOption>,
        O::IntoIter: ExactSizeIterator,
    {
        self.canonical_functions().future_read(ty, options);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `future.write` intrinsic.
    pub fn future_write<O>(&mut self, ty: u32, options: O) -> u32
    where
        O: IntoIterator<Item = CanonicalOption>,
        O::IntoIter: ExactSizeIterator,
    {
        self.canonical_functions().future_write(ty, options);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `future.cancel-read` intrinsic.
    pub fn future_cancel_read(&mut self, ty: u32, async_: bool) -> u32 {
        self.canonical_functions().future_cancel_read(ty, async_);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `future.cancel-write` intrinsic.
    pub fn future_cancel_write(&mut self, ty: u32, async_: bool) -> u32 {
        self.canonical_functions().future_cancel_read(ty, async_);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `future.close-readable` intrinsic.
    pub fn future_close_readable(&mut self, ty: u32) -> u32 {
        self.canonical_functions().future_close_readable(ty);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `future.close-writable` intrinsic.
    pub fn future_close_writable(&mut self, ty: u32) -> u32 {
        self.canonical_functions().future_close_writable(ty);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `error.new` intrinsic.
    pub fn error_new<O>(&mut self, options: O) -> u32
    where
        O: IntoIterator<Item = CanonicalOption>,
        O::IntoIter: ExactSizeIterator,
    {
        self.canonical_functions().error_new(options);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `error.debug-message` intrinsic.
    pub fn error_debug_message<O>(&mut self, options: O) -> u32
    where
        O: IntoIterator<Item = CanonicalOption>,
        O::IntoIter: ExactSizeIterator,
    {
        self.canonical_functions().error_debug_message(options);
        inc(&mut self.core_funcs)
    }

    /// Declares a new `error.drop` intrinsic.
    pub fn error_drop(&mut self) -> u32 {
        self.canonical_functions().error_drop();
        inc(&mut self.core_funcs)
    }

    /// Adds a new custom section to this component.
    pub fn custom_section(&mut self, section: &CustomSection<'_>) {
        self.flush();
        self.component.section(section);
    }

    /// Adds a new custom section to this component.
    pub fn raw_custom_section(&mut self, section: &[u8]) {
        self.flush();
        self.component.section(&RawCustomSection(section));
    }
}

// Helper macro to generate methods on `ComponentBuilder` to get specific
// section encoders that automatically flush and write out prior sections as
// necessary.
macro_rules! section_accessors {
    ($($method:ident => $section:ident)*) => (
        #[derive(Debug, Default)]
        enum LastSection {
            #[default]
            None,
            $($section($section),)*
        }

        impl ComponentBuilder {
            $(
                fn $method(&mut self) -> &mut $section {
                    match &self.last_section {
                        // The last encoded section matches the section that's
                        // being requested, so no change is necessary.
                        LastSection::$section(_) => {}

                        // Otherwise the last section didn't match this section,
                        // so flush any prior section if needed and start
                        // encoding the desired section of this method.
                        _ => {
                            self.flush();
                            self.last_section = LastSection::$section($section::new());
                        }
                    }
                    match &mut self.last_section {
                        LastSection::$section(ret) => ret,
                        _ => unreachable!()
                    }
                }
            )*

            /// Writes out the last section into the final component binary if
            /// there is a section specified, otherwise does nothing.
            fn flush(&mut self) {
                match mem::take(&mut self.last_section) {
                    LastSection::None => {}
                    $(
                        LastSection::$section(section) => {
                            self.component.section(&section);
                        }
                    )*
                }
            }

        }
    )
}

section_accessors! {
    component_instances => ComponentInstanceSection
    instances => InstanceSection
    canonical_functions => CanonicalFunctionSection
    aliases => ComponentAliasSection
    exports => ComponentExportSection
    imports => ComponentImportSection
    types => ComponentTypeSection
    core_types => CoreTypeSection
}

fn inc(idx: &mut u32) -> u32 {
    let ret = *idx;
    *idx += 1;
    ret
}
