use crate::encoding::{Instance, Item, LibraryInfo, MainOrAdapter};
use crate::ComponentEncoder;
use anyhow::{bail, Context, Result};
use indexmap::{map::Entry, IndexMap, IndexSet};
use std::hash::{Hash, Hasher};
use std::mem;
use wasm_encoder::ExportKind;
use wasmparser::names::{ComponentName, ComponentNameKind};
use wasmparser::{
    types::TypesRef, Encoding, ExternalKind, FuncType, Parser, Payload, TypeRef, ValType,
    ValidPayload, Validator,
};
use wit_parser::{
    abi::{AbiVariant, WasmSignature, WasmType},
    Function, InterfaceId, PackageName, Resolve, TypeDefKind, TypeId, World, WorldId, WorldItem,
    WorldKey,
};

fn wasm_sig_to_func_type(signature: WasmSignature) -> FuncType {
    fn from_wasm_type(ty: &WasmType) -> ValType {
        match ty {
            WasmType::I32 => ValType::I32,
            WasmType::I64 => ValType::I64,
            WasmType::F32 => ValType::F32,
            WasmType::F64 => ValType::F64,
            WasmType::Pointer => ValType::I32,
            WasmType::PointerOrI64 => ValType::I64,
            WasmType::Length => ValType::I32,
        }
    }

    FuncType::new(
        signature.params.iter().map(from_wasm_type),
        signature.results.iter().map(from_wasm_type),
    )
}

pub const MAIN_MODULE_IMPORT_NAME: &str = "__main_module__";

/// The module name used when a top-level function in a world is imported into a
/// core wasm module. Note that this is not a valid WIT identifier to avoid
/// clashes with valid WIT interfaces. This is also not empty because LLVM
/// interprets an empty module import string as "not specified" which means it
/// turns into `env`.
pub const BARE_FUNC_MODULE_NAME: &str = "$root";

pub const RESOURCE_DROP: &str = "[resource-drop]";
pub const RESOURCE_REP: &str = "[resource-rep]";
pub const RESOURCE_NEW: &str = "[resource-new]";

pub const ASYNC_START: &str = "[async-start]";
pub const ASYNC_RETURN: &str = "[async-return]";

pub const POST_RETURN_PREFIX: &str = "cabi_post_";
pub const CALLBACK_PREFIX: &str = "[callback][async]";

/// Metadata about a validated module and what was found internally.
///
/// All imports to the module are described by the union of `required_imports`
/// and `adapters_required`.
///
/// This structure is created by the `validate_module` function.
pub struct ValidatedModule<'a> {
    /// The required imports into this module which are to be satisfied by
    /// imported component model instances.
    ///
    /// The key of this map is the name of the interface that the module imports
    /// from and the value is the set of functions required from that interface.
    /// This is used to generate an appropriate instance import in the generated
    /// component which imports only the set of required functions.
    pub required_imports: IndexMap<&'a str, RequiredImports>,

    /// This is the set of imports into the module which were not satisfied by
    /// imported interfaces but are required to be satisfied by adapter modules.
    ///
    /// The key of this map is the name of the adapter that was imported into
    /// the module and the value is a further map from function to function type
    /// as required by this module. This map is used to shrink adapter modules
    /// to the precise size required for this module by ensuring it doesn't
    /// export (and subsequently import) extraneous functions.
    pub adapters_required: IndexMap<&'a str, IndexMap<&'a str, FuncType>>,

    /// Resource-related functions required and imported which work over
    /// exported resources from the final component.
    ///
    /// Note that this is disjoint from `required_imports` which handles
    /// imported resources and this is only for exported resources. Exported
    /// resources still require intrinsics to be imported into the core module
    /// itself.
    pub required_resource_funcs: IndexMap<String, IndexMap<String, ResourceInfo>>,

    pub required_async_funcs: IndexMap<String, IndexMap<String, AsyncExportInfo<'a>>>,

    pub required_payload_funcs:
        IndexMap<(String, bool), IndexMap<(String, usize), PayloadInfo<'a>>>,

    pub needs_error_drop: bool,
    pub needs_task_wait: bool,

    /// Whether or not this module exported a linear memory.
    pub has_memory: bool,

    /// Whether or not this module exported a `cabi_realloc` function.
    pub realloc: Option<&'a str>,

    /// Whether or not this module exported a `cabi_realloc_adapter` function.
    pub adapter_realloc: Option<&'a str>,

    /// The original metadata specified for this module.
    pub metadata: &'a ModuleMetadata,

    /// Post-return functions annotated with `cabi_post_*` in their function
    /// name.
    pub post_returns: IndexSet<String>,

    /// Callback functions annotated with `[callback]*` in their function
    /// name.
    pub callbacks: IndexSet<String>,

    /// Exported function like `_initialize` which needs to be run after
    /// everything else has been instantiated.
    pub initialize: Option<&'a str>,
}

/// Metadata about a validated module and what was found internally.
///
/// This structure houses information about `imports` and `exports` to the
/// module. Each of these specialized types contains "connection" information
/// between a module's imports/exports and the WIT or component-level constructs
/// they correspond to.

#[derive(Default)]
pub struct ValidatedModule {
    /// Information about a module's imports.
    pub imports: ImportMap,

    /// Information about a module's exports.
    pub exports: ExportMap,
}

impl ValidatedModule {
    fn new(
        encoder: &ComponentEncoder,
        bytes: &[u8],
        exports: &IndexSet<WorldKey>,
        info: Option<&LibraryInfo>,
    ) -> Result<ValidatedModule> {
        let mut validator = Validator::new();
        let mut ret = ValidatedModule::default();

        for payload in Parser::new(0).parse_all(bytes) {
            let payload = payload?;
            if let ValidPayload::End(_) = validator.payload(&payload)? {
                break;
            }

            let types = validator.types(0).unwrap();

            match payload {
                Payload::Version { encoding, .. } if encoding != Encoding::Module => {
                    bail!("data is not a WebAssembly module");
                }
                Payload::ImportSection(s) => {
                    for import in s {
                        let import = import?;
                        ret.imports.add(import, encoder, info, types)?;
                    }
                }
                Payload::ExportSection(s) => {
                    for export in s {
                        let export = export?;
                        ret.exports.add(export, encoder, &exports, types)?;
                    }
                }
                _ => continue,
            }
        }

        ret.exports.validate(encoder, exports)?;

        Ok(ret)
    }
}

/// Metadata information about a module's imports.
///
/// This structure maintains the connection between component model "things" and
/// core wasm "things" by ensuring that all imports to the core wasm module are
/// classified by the `Import` enumeration.
#[derive(Default)]
pub struct ImportMap {
    /// The first level of the map here is the module namespace of the import
    /// and the second level of the map is the field namespace. The item is then
    /// how the import is satisfied.
    names: IndexMap<String, ImportInstance>,
}

pub enum ImportInstance {
    /// This import is satisfied by an entire instance of another
    /// adapter/module.
    Whole(MainOrAdapter),

    /// This import is satisfied by filling out each name possibly differently.
    Names(IndexMap<String, Import>),
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct PayloadInfo {
    pub name: String,
    pub ty: TypeId,
    pub function: Function,
    pub key: WorldKey,
    pub interface: Option<InterfaceId>,
    pub imported: bool,
}

impl Hash for PayloadInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.ty.hash(state);
        self.key.hash(state);
        self.interface.hash(state);
        self.imported.hash(state);
    }
}

/// The different kinds of items that a module or an adapter can import.
///
/// This is intended to be an exhaustive definition of what can be imported into
/// core modules within a component that wit-component supports.
#[derive(Debug, Clone)]
pub enum Import {
    /// A top-level world function, with the name provided here, is imported
    /// into the module.
    WorldFunc(WorldKey, String, AbiVariant),

    /// An interface's function is imported into the module.
    ///
    /// The `WorldKey` here is the name of the interface in the world in
    /// question. The `InterfaceId` is the interface that was imported from and
    /// `String` is the WIT name of the function.
    InterfaceFunc(WorldKey, InterfaceId, String, AbiVariant),

    /// An imported resource's destructor is imported.
    ///
    /// The key provided indicates whether it's for the top-level types of the
    /// world (`None`) or an interface (`Some` with the name of the interface).
    /// The `TypeId` is what resource is being dropped.
    ImportedResourceDrop(WorldKey, Option<InterfaceId>, TypeId),

    /// A `canon resource.drop` intrinsic for an exported item is being
    /// imported.
    ///
    /// This lists the key of the interface that's exporting the resource plus
    /// the id within that interface.
    ExportedResourceDrop(WorldKey, TypeId),

    /// A `canon resource.new` intrinsic for an exported item is being
    /// imported.
    ///
    /// This lists the key of the interface that's exporting the resource plus
    /// the id within that interface.
    ExportedResourceNew(WorldKey, TypeId),

    /// A `canon resource.rep` intrinsic for an exported item is being
    /// imported.
    ///
    /// This lists the key of the interface that's exporting the resource plus
    /// the id within that interface.
    ExportedResourceRep(WorldKey, TypeId),

    /// An export of an adapter is being imported with the specified type.
    ///
    /// This is used for when the main module imports an adapter function. The
    /// adapter name and function name match the module's own import, and the
    /// type must match that listed here.
    AdapterExport(FuncType),

    /// An adapter is importing the memory of the main module.
    ///
    /// (should be combined with `MainModuleExport` below one day)
    MainModuleMemory,

    /// An adapter is importing an arbitrary item from the main module.
    MainModuleExport {
        name: String,
        kind: ExportKind,
    },

    /// An arbitrary item from either the main module or an adapter is being
    /// imported.
    ///
    /// (should probably subsume `MainModule*` and maybe `AdapterExport` above
    /// one day.
    Item(Item),

    ErrorDrop,
    TaskBackpressure,
    TaskWait,
    TaskPoll,
    TaskYield,
    SubtaskDrop,
    FutureNew(PayloadInfo),
    FutureWrite(PayloadInfo),
    FutureRead(PayloadInfo),
    FutureDropWriter(TypeId),
    FutureDropReader(TypeId),
    StreamNew(PayloadInfo),
    StreamWrite(PayloadInfo),
    StreamRead(PayloadInfo),
    StreamDropWriter(TypeId),
    StreamDropReader(TypeId),
    ExportedTaskReturn(Function),
}

impl ImportMap {
    /// Returns whether the top-level world function `func` is imported.
    pub fn uses_toplevel_func(&self, func: &str) -> bool {
        self.imports().any(|(_, _, item)| match item {
            Import::WorldFunc(_, name, _) => func == name,
            _ => false,
        })
    }

    /// Returns whether the interface function specified is imported.
    pub fn uses_interface_func(&self, interface: InterfaceId, func: &str) -> bool {
        self.imports().any(|(_, _, import)| match import {
            Import::InterfaceFunc(_, id, name, _) => *id == interface && name == func,
            _ => false,
        })
    }

    /// Returns whether the specified resource's drop method is needed to import.
    pub fn uses_imported_resource_drop(&self, resource: TypeId) -> bool {
        self.imports().any(|(_, _, import)| match import {
            Import::ImportedResourceDrop(_, _, id) => resource == *id,
            _ => false,
        })
    }

    /// Returns the list of items that the adapter named `name` must export.
    pub fn required_from_adapter(&self, name: &str) -> IndexMap<String, FuncType> {
        let names = match self.names.get(name) {
            Some(ImportInstance::Names(names)) => names,
            _ => return IndexMap::new(),
        };
        names
            .iter()
            .map(|(name, import)| {
                (
                    name.clone(),
                    match import {
                        Import::AdapterExport(ty) => ty.clone(),
                        _ => unreachable!(),
                    },
                )
            })
            .collect()
    }

    /// Returns an iterator over all individual imports registered in this map.
    ///
    /// Note that this doesn't iterate over the "whole instance" imports.
    pub fn imports(&self) -> impl Iterator<Item = (&str, &str, &Import)> + '_ {
        self.names
            .iter()
            .filter_map(|(module, m)| match m {
                ImportInstance::Names(names) => Some((module, names)),
                ImportInstance::Whole(_) => None,
            })
            .flat_map(|(module, m)| {
                m.iter()
                    .map(move |(field, import)| (module.as_str(), field.as_str(), import))
            })
    }

    /// Returns the map for how all imports must be satisfied.
    pub fn modules(&self) -> &IndexMap<String, ImportInstance> {
        &self.names
    }

    /// Helper function used during validation to build up this `ImportMap`.
    fn add(
        &mut self,
        import: wasmparser::Import<'_>,
        encoder: &ComponentEncoder,
        library_info: Option<&LibraryInfo>,
        types: TypesRef<'_>,
    ) -> Result<()> {
        if self.classify_import_with_library(import, library_info)? {
            return Ok(());
        }
        let item = self.classify(import, encoder, types).with_context(|| {
            format!(
                "failed to resolve import `{}::{}`",
                import.module, import.name,
            )
        })?;
        self.insert_import(import, item)
    }

    fn classify(
        &self,
        import: wasmparser::Import<'_>,
        encoder: &ComponentEncoder,
        types: TypesRef<'_>,
    ) -> Result<Import> {
        // Special-case the main module's memory imported into adapters which
        // currently with `wasm-ld` is not easily configurable.
        if import.module == "env" && import.name == "memory" {
            return Ok(Import::MainModuleMemory);
        }

        // Special-case imports from the main module into adapters.
        if import.module == "__main_module__" {
            return Ok(Import::MainModuleExport {
                name: import.name.to_string(),
                kind: match import.ty {
                    TypeRef::Func(_) => ExportKind::Func,
                    TypeRef::Table(_) => ExportKind::Table,
                    TypeRef::Memory(_) => ExportKind::Memory,
                    TypeRef::Global(_) => ExportKind::Global,
                    TypeRef::Tag(_) => ExportKind::Tag,
                },
            });
        }

        let ty_index = match import.ty {
            TypeRef::Func(ty) => ty,
            _ => bail!("module is only allowed to import functions"),
        };
        let ty = types[types.core_type_at_in_module(ty_index)].unwrap_func();

        // Handle main module imports that match known adapters and set it up as
        // an import of an adapter export.
        if encoder.adapters.contains_key(import.module) {
            return Ok(Import::AdapterExport(ty.clone()));
        }

        let (module, names) = match import.module.strip_prefix("cm32p2") {
            Some(suffix) => (suffix, STANDARD),
            None if encoder.reject_legacy_names => (import.module, STANDARD),
            None => (import.module, LEGACY),
        };
        self.classify_component_model_import(module, import.name, encoder, ty, names)
    }

    /// Attempts to classify the import `{module}::{name}` with the rules
    /// specified in WebAssembly/component-model#378
    fn classify_component_model_import(
        &self,
        module: &str,
        name: &str,
        encoder: &ComponentEncoder,
        ty: &FuncType,
        names: &dyn NameMangling,
    ) -> Result<Import> {
        let resolve = &encoder.metadata.resolve;
        let world_id = encoder.metadata.world;
        let world = &resolve.worlds[world_id];

        if let Some(import) = names.payload_import(module, name, resolve, world, ty)? {
            return Ok(import);
        }

        let async_import = |interface: Option<(WorldKey, InterfaceId)>| {
            Ok::<_, anyhow::Error>(if let Some(function_name) = names.task_return_name(name) {
                let interface_id = interface.as_ref().map(|(_, id)| *id);
                let func = get_function(resolve, world, function_name, interface_id, true)?;
                validate_task_return(resolve, ty, &func)?;
                Some(Import::ExportedTaskReturn(func))
            } else {
                None
            })
        };

        let (abi, name) = if let Some(name) = names.async_name(name) {
            (AbiVariant::GuestImportAsync, name)
        } else {
            (AbiVariant::GuestImport, name)
        };

        if module == names.import_root() {
            if Some(name) == names.error_drop() {
                let expected = FuncType::new([ValType::I32], []);
                validate_func_sig(name, &expected, ty)?;
                return Ok(Import::ErrorDrop);
            }

            if Some(name) == names.task_backpressure() {
                let expected = FuncType::new([ValType::I32], []);
                validate_func_sig(name, &expected, ty)?;
                return Ok(Import::TaskBackpressure);
            }

            if Some(name) == names.task_wait() {
                let expected = FuncType::new([ValType::I32], [ValType::I32]);
                validate_func_sig(name, &expected, ty)?;
                return Ok(Import::TaskWait);
            }

            if Some(name) == names.task_poll() {
                let expected = FuncType::new([ValType::I32], [ValType::I32]);
                validate_func_sig(name, &expected, ty)?;
                return Ok(Import::TaskPoll);
            }

            if Some(name) == names.task_yield() {
                let expected = FuncType::new([], []);
                validate_func_sig(name, &expected, ty)?;
                return Ok(Import::TaskYield);
            }

            if Some(name) == names.subtask_drop() {
                let expected = FuncType::new([ValType::I32], []);
                validate_func_sig(name, &expected, ty)?;
                return Ok(Import::SubtaskDrop);
            }

            if let Some(import) = async_import(None)? {
                return Ok(import);
            }

            let key = WorldKey::Name(name.to_string());
            if let Some(WorldItem::Function(func)) = world.imports.get(&key) {
                validate_func(resolve, ty, func, abi)?;
                return Ok(Import::WorldFunc(key, func.name.clone(), abi));
            }

            let get_resource = resource_test_for_world(resolve, world_id);
            if let Some(resource) = names.resource_drop_name(name) {
                if let Some(id) = get_resource(resource) {
                    let expected = FuncType::new([ValType::I32], []);
                    validate_func_sig(name, &expected, ty)?;
                    return Ok(Import::ImportedResourceDrop(key, None, id));
                }
            }

            match world.imports.get(&key) {
                Some(_) => bail!("expected world top-level import `{name}` to be a function"),
                None => bail!("no top-level imported function `{name}` specified"),
            }
        }

        let interface = match module.strip_prefix(names.import_non_root_prefix()) {
            Some(name) => name,
            None => bail!("unknown or invalid component model import syntax"),
        };

        if let Some(interface) = interface.strip_prefix(names.import_exported_intrinsic_prefix()) {
            if let Some(import) = async_import(Some(names.module_to_interface(
                interface,
                resolve,
                &world.exports,
            )?))? {
                return Ok(import);
            }

            let (key, id) = names.module_to_interface(interface, resolve, &world.exports)?;

            let get_resource = resource_test_for_interface(resolve, id);
            if let Some(name) = names.resource_drop_name(name) {
                if let Some(id) = get_resource(name) {
                    let expected = FuncType::new([ValType::I32], []);
                    validate_func_sig(name, &expected, ty)?;
                    return Ok(Import::ExportedResourceDrop(key, id));
                }
            }
            if let Some(name) = names.resource_new_name(name) {
                if let Some(id) = get_resource(name) {
                    let expected = FuncType::new([ValType::I32], [ValType::I32]);
                    validate_func_sig(name, &expected, ty)?;
                    return Ok(Import::ExportedResourceNew(key, id));
                }
            }
            if let Some(name) = names.resource_rep_name(name) {
                if let Some(id) = get_resource(name) {
                    let expected = FuncType::new([ValType::I32], [ValType::I32]);
                    validate_func_sig(name, &expected, ty)?;
                    return Ok(Import::ExportedResourceRep(key, id));
                }
            }
            bail!("unknown function `{name}`")
        }

        let (key, id) = names.module_to_interface(interface, resolve, &world.imports)?;
        let interface = &resolve.interfaces[id];
        let get_resource = resource_test_for_interface(resolve, id);
        if let Some(f) = interface.functions.get(name) {
            validate_func(resolve, ty, f, abi).with_context(|| {
                let name = resolve.name_world_key(&key);
                format!("failed to validate import interface `{name}`")
            })?;
            return Ok(Import::InterfaceFunc(key, id, f.name.clone(), abi));
        } else if let Some(resource) = names.resource_drop_name(name) {
            if let Some(resource) = get_resource(resource) {
                let expected = FuncType::new([ValType::I32], []);
                validate_func_sig(name, &expected, ty)?;
                return Ok(Import::ImportedResourceDrop(key, Some(id), resource));
            }
        }
        bail!(
            "import interface `{module}` is missing function \
             `{name}` that is required by the module",
        )
    }

    fn classify_import_with_library(
        &mut self,
        import: wasmparser::Import<'_>,
        library_info: Option<&LibraryInfo>,
    ) -> Result<bool> {
        let info = match library_info {
            Some(info) => info,
            None => return Ok(false),
        };
        let Some((_, instance)) = info
            .arguments
            .iter()
            .find(|(name, _items)| *name == import.module)
        else {
            return Ok(false);
        };
        match instance {
            Instance::MainOrAdapter(module) => match self.names.get(import.module) {
                Some(ImportInstance::Whole(which)) => {
                    if which != module {
                        bail!("different whole modules imported under the same name");
                    }
                }
                Some(ImportInstance::Names(_)) => {
                    bail!("cannot mix individual imports and whole module imports")
                }
                None => {
                    let instance = ImportInstance::Whole(module.clone());
                    self.names.insert(import.module.to_string(), instance);
                }
            },
            Instance::Items(items) => {
                let Some(item) = items.iter().find(|i| i.alias == import.name) else {
                    return Ok(false);
                };
                self.insert_import(import, Import::Item(item.clone()))?;
            }
        }
        Ok(true)
    }

    fn insert_import(&mut self, import: wasmparser::Import<'_>, item: Import) -> Result<()> {
        let entry = self
            .names
            .entry(import.module.to_string())
            .or_insert(ImportInstance::Names(IndexMap::default()));
        let names = match entry {
            ImportInstance::Names(names) => names,
            _ => bail!("cannot mix individual imports with module imports"),
        };
        let entry = match names.entry(import.name.to_string()) {
            Entry::Occupied(_) => {
                bail!(
                    "module has duplicate import for `{}::{}`",
                    import.module,
                    import.name
                );
            }
            Entry::Vacant(v) => v,
        };
        log::trace!(
            "classifying import `{}::{} as {item:?}",
            import.module,
            import.name
        );
        entry.insert(item);
        Ok(())
    }
}

/// Dual of `ImportMap` except describes the exports of a module instead of the
/// imports.
#[derive(Default)]
pub struct ExportMap {
    names: IndexMap<String, Export>,
    raw_exports: IndexMap<String, FuncType>,
}

/// All possible (known) exports from a core wasm module that are recognized and
/// handled during the componentization process.
#[derive(Debug)]
pub enum Export {
    /// An export of a top-level function of a world, where the world function
    /// is named here.
    WorldFunc(WorldKey, String, AbiVariant),

    /// A post-return for a top-level function of a world.
    WorldFuncPostReturn(WorldKey),

    /// An export of a function in an interface.
    InterfaceFunc(WorldKey, InterfaceId, String, AbiVariant),

    /// A post-return for the above function.
    InterfaceFuncPostReturn(WorldKey, String),

    /// A destructor for an exported resource.
    ResourceDtor(TypeId),

    /// Memory, typically for an adapter.
    Memory,

    /// `cabi_realloc`
    GeneralPurposeRealloc,

    /// `cabi_export_realloc`
    GeneralPurposeExportRealloc,

    /// `cabi_import_realloc`
    GeneralPurposeImportRealloc,

    /// `_initialize`
    Initialize,

    /// `cabi_realloc_adapter`
    ReallocForAdapter,

    WorldFuncCallback(WorldKey),

    InterfaceFuncCallback(WorldKey, String),
}

impl ExportMap {
    fn add(
        &mut self,
        export: wasmparser::Export<'_>,
        encoder: &ComponentEncoder,
        exports: &IndexSet<WorldKey>,
        types: TypesRef<'_>,
    ) -> Result<()> {
        if let Some(item) = self.classify(export, encoder, exports, types)? {
            log::debug!("classifying export `{}` as {item:?}", export.name);
            let prev = self.names.insert(export.name.to_string(), item);
            assert!(prev.is_none());
        }
        Ok(())
    }

    fn classify(
        &mut self,
        export: wasmparser::Export<'_>,
        encoder: &ComponentEncoder,
        exports: &IndexSet<WorldKey>,
        types: TypesRef<'_>,
    ) -> Result<Option<Export>> {
        match export.kind {
            ExternalKind::Func => {
                let ty = types[types.core_function_at(export.index)].unwrap_func();
                self.raw_exports.insert(export.name.to_string(), ty.clone());
            }
            _ => {}
        }

        // Handle a few special-cased names first.
        if export.name == "canonical_abi_realloc" {
            return Ok(Some(Export::GeneralPurposeRealloc));
        } else if export.name == "cabi_import_realloc" {
            return Ok(Some(Export::GeneralPurposeImportRealloc));
        } else if export.name == "cabi_export_realloc" {
            return Ok(Some(Export::GeneralPurposeExportRealloc));
        } else if export.name == "cabi_realloc_adapter" {
            return Ok(Some(Export::ReallocForAdapter));
        }

        let (name, names) = match export.name.strip_prefix("cm32p2") {
            Some(name) => (name, STANDARD),
            None if encoder.reject_legacy_names => return Ok(None),
            None => (export.name, LEGACY),
        };
        if let Some(export) = self
            .classify_component_export(names, name, &export, encoder, exports, types)
            .with_context(|| format!("failed to classify export `{}`", export.name))?
        {
            return Ok(Some(export));
        }
        log::debug!("unknown export `{}`", export.name);
        Ok(None)
    }

    fn classify_component_export(
        &mut self,
        names: &dyn NameMangling,
        name: &str,
        export: &wasmparser::Export<'_>,
        encoder: &ComponentEncoder,
        exports: &IndexSet<WorldKey>,
        types: TypesRef<'_>,
    ) -> Result<Option<Export>> {
        let resolve = &encoder.metadata.resolve;
        let world = encoder.metadata.world;
        match export.kind {
            ExternalKind::Func => {}
            ExternalKind::Memory => {
                if name == names.export_memory() {
                    return Ok(Some(Export::Memory));
                }
                return Ok(None);
            }
            _ => return Ok(None),
        }
        let ty = types[types.core_function_at(export.index)].unwrap_func();

        // Handle a few special-cased names first.
        if name == names.export_realloc() {
            let expected = FuncType::new([ValType::I32; 4], [ValType::I32]);
            validate_func_sig(name, &expected, ty)?;
            return Ok(Some(Export::GeneralPurposeRealloc));
        } else if name == names.export_initialize() {
            let expected = FuncType::new([], []);
            validate_func_sig(name, &expected, ty)?;
            return Ok(Some(Export::Initialize));
        }

        let (abi, name) = if let Some(name) = names.async_name(export.name) {
            (AbiVariant::GuestExportAsync, name)
        } else {
            (AbiVariant::GuestExport, export.name)
        };

        // Try to match this to a known WIT export that `exports` allows.
        if let Some((key, id, f)) = names.match_wit_export(name, resolve, world, exports) {
            validate_func(resolve, ty, f, abi).with_context(|| {
                let key = resolve.name_world_key(key);
                format!("failed to validate export for `{key}`")
            })?;
            match id {
                Some(id) => {
                    return Ok(Some(Export::InterfaceFunc(
                        key.clone(),
                        id,
                        f.name.clone(),
                        abi,
                    )));
                }
                None => {
                    return Ok(Some(Export::WorldFunc(key.clone(), f.name.clone(), abi)));
                }
            }
        }

        // See if this is a post-return for any known WIT export.
        if let Some(remaining) = names.strip_post_return(name) {
            if let Some((key, id, f)) = names.match_wit_export(remaining, resolve, world, exports) {
                validate_post_return(resolve, ty, f).with_context(|| {
                    let key = resolve.name_world_key(key);
                    format!("failed to validate export for `{key}`")
                })?;
                match id {
                    Some(_id) => {
                        return Ok(Some(Export::InterfaceFuncPostReturn(
                            key.clone(),
                            f.name.clone(),
                        )));
                    }
                    None => {
                        return Ok(Some(Export::WorldFuncPostReturn(key.clone())));
                    }
                }
            }
        }

        if let Some(suffix) = names.callback_name(export.name) {
            if let Some((key, id, f)) = names.match_wit_export(suffix, resolve, world, exports) {
                validate_func_sig(
                    export.name,
                    &FuncType::new([ValType::I32; 4], [ValType::I32]),
                    ty,
                )?;
                return Ok(Some(if id.is_some() {
                    Export::InterfaceFuncCallback(key.clone(), f.name.clone())
                } else {
                    Export::WorldFuncCallback(key.clone())
                }));
            }
        }

        // And, finally, see if it matches a known destructor.
        if let Some(dtor) = names.match_wit_resource_dtor(name, resolve, world, exports) {
            let expected = FuncType::new([ValType::I32], []);
            validate_func_sig(export.name, &expected, ty)?;
            return Ok(Some(Export::ResourceDtor(dtor)));
        }

        Ok(None)
    }

    /// Returns the name of the post-return export, if any, for the `key` and
    /// `func` combo.
    pub fn post_return(&self, key: &WorldKey, func: &Function) -> Option<&str> {
        self.find(|m| match m {
            Export::WorldFuncPostReturn(k) => k == key,
            Export::InterfaceFuncPostReturn(k, f) => k == key && func.name == *f,
            _ => false,
        })
    }

    /// Returns the name of the async callback export, if any, for the `key` and
    /// `func` combo.
    pub fn callback(&self, key: &WorldKey, func: &Function) -> Option<&str> {
        self.find(|m| match m {
            Export::WorldFuncCallback(k) => k == key,
            Export::InterfaceFuncCallback(k, f) => k == key && func.name == *f,
            _ => false,
        })
    }

    pub fn abi(&self, key: &WorldKey, func: &Function) -> Option<AbiVariant> {
        self.names.values().find_map(|m| match m {
            Export::WorldFunc(k, f, abi) if k == key && func.name == *f => Some(*abi),
            Export::InterfaceFunc(k, _, f, abi) if k == key && func.name == *f => Some(*abi),
            _ => None,
        })
    }

    /// Returns the realloc that the exported function `interface` and `func`
    /// are using.
    pub fn export_realloc_for(&self, key: &WorldKey, func: &Function) -> Option<&str> {
        // TODO: This realloc detection should probably be improved with
        // some sort of scheme to have per-function reallocs like
        // `cabi_realloc_{name}` or something like that.
        let _ = (key, func);

        if let Some(name) = self.find(|m| matches!(m, Export::GeneralPurposeExportRealloc)) {
            return Some(name);
        }
        self.general_purpose_realloc()
    }

    /// Returns the realloc that the imported function `interface` and `func`
    /// are using.
    pub fn import_realloc_for(&self, interface: Option<InterfaceId>, func: &str) -> Option<&str> {
        // TODO: This realloc detection should probably be improved with
        // some sort of scheme to have per-function reallocs like
        // `cabi_realloc_{name}` or something like that.
        let _ = (interface, func);

        if let Some(name) = self.find(|m| matches!(m, Export::GeneralPurposeImportRealloc)) {
            return Some(name);
        }
        self.general_purpose_realloc()
    }

    /// Returns the realloc that the main module is exporting into the adapter.
    pub fn realloc_to_import_into_adapter(&self) -> Option<&str> {
        if let Some(name) = self.find(|m| matches!(m, Export::ReallocForAdapter)) {
            return Some(name);
        }
        self.general_purpose_realloc()
    }

    fn general_purpose_realloc(&self) -> Option<&str> {
        self.find(|m| matches!(m, Export::GeneralPurposeRealloc))
    }

    /// Returns the memory, if exported, for this module.
    pub fn memory(&self) -> Option<&str> {
        self.find(|m| matches!(m, Export::Memory))
    }

    /// Returns the `_initialize` intrinsic, if exported, for this module.
    pub fn initialize(&self) -> Option<&str> {
        self.find(|m| matches!(m, Export::Initialize))
    }

    /// Returns destructor for the exported resource `ty`, if it was listed.
    pub fn resource_dtor(&self, ty: TypeId) -> Option<&str> {
        self.find(|m| match m {
            Export::ResourceDtor(t) => *t == ty,
            _ => false,
        })
    }

    /// NB: this is a linear search and if that's ever a problem this should
    /// build up an inverse map during construction to accelerate it.
    fn find(&self, f: impl Fn(&Export) -> bool) -> Option<&str> {
        let (name, _) = self.names.iter().filter(|(_, m)| f(m)).next()?;
        Some(name)
    }

    /// Iterates over all exports of this module.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Export)> + '_ {
        self.names.iter().map(|(n, e)| (n.as_str(), e))
    }

    fn validate(&self, encoder: &ComponentEncoder, exports: &IndexSet<WorldKey>) -> Result<()> {
        let resolve = &encoder.metadata.resolve;
        let world = encoder.metadata.world;
        // Multi-memory isn't supported because otherwise we don't know what
        // memory to put things in.
        if self
            .names
            .values()
            .filter(|m| matches!(m, Export::Memory))
            .count()
            > 1
        {
            bail!("cannot componentize module that exports multiple memories")
        }

        // All of `exports` must be exported and found within this module.
        for export in exports {
            let require_interface_func = |interface: InterfaceId, name: &str| -> Result<()> {
                let result = self.find(|e| match e {
                    Export::InterfaceFunc(_, id, s, _) => interface == *id && name == s,
                    _ => false,
                });
                if result.is_some() {
                    Ok(())
                } else {
                    let export = resolve.name_world_key(export);
                    bail!("failed to find export of interface `{export}` function `{name}`")
                }
            };
            let require_world_func = |name: &str| -> Result<()> {
                let result = self.find(|e| match e {
                    Export::WorldFunc(_, s, _) => name == s,
                    _ => false,
                });
                if result.is_some() {
                    Ok(())
                } else {
                    bail!("failed to find export of function `{name}`")
                }
            };
            match &resolve.worlds[world].exports[export] {
                WorldItem::Interface { id, .. } => {
                    for (name, _) in resolve.interfaces[*id].functions.iter() {
                        require_interface_func(*id, name)?;
                    }
                }
                WorldItem::Function(f) => {
                    require_world_func(&f.name)?;
                }
                WorldItem::Type(_) => unreachable!(),
            }
        }

        Ok(())
    }
}

/// Trait dispatch and definition for parsing and interpreting "mangled names"
/// which show up in imports and exports of the component model.
///
/// This trait is used to implement classification of imports and exports in the
/// component model. The methods on `ImportMap` and `ExportMap` will use this to
/// determine what an import is and how it's lifted/lowered in the world being
/// bound.
///
/// This trait has a bit of history behind it as well. Before
/// WebAssembly/component-model#378 there was no standard naming scheme for core
/// wasm imports or exports when componenitizing. This meant that
/// `wit-component` implemented a particular scheme which mostly worked but was
/// mostly along the lines of "this at least works" rather than "someone sat
/// down and designed this". Since then, however, an standard naming scheme has
/// now been specified which was indeed designed.
///
/// This trait serves as the bridge between these two. The historical naming
/// scheme is still supported for now through the `Legacy` implementation below
/// and will be for some time. The transition plan at this time is to support
/// the new scheme, eventually get it supported in bindings generators, and once
/// that's all propagated remove support for the legacy scheme.
trait NameMangling {
    fn import_root(&self) -> &str;
    fn import_non_root_prefix(&self) -> &str;
    fn import_exported_intrinsic_prefix(&self) -> &str;
    fn export_memory(&self) -> &str;
    fn export_initialize(&self) -> &str;
    fn export_realloc(&self) -> &str;
    fn resource_drop_name<'a>(&self, s: &'a str) -> Option<&'a str>;
    fn resource_new_name<'a>(&self, s: &'a str) -> Option<&'a str>;
    fn resource_rep_name<'a>(&self, s: &'a str) -> Option<&'a str>;
    fn task_return_name<'a>(&self, s: &'a str) -> Option<&'a str>;
    fn task_backpressure(&self) -> Option<&str>;
    fn task_wait(&self) -> Option<&str>;
    fn task_poll(&self) -> Option<&str>;
    fn task_yield(&self) -> Option<&str>;
    fn subtask_drop(&self) -> Option<&str>;
    fn callback_name<'a>(&self, s: &'a str) -> Option<&'a str>;
    fn async_name<'a>(&self, s: &'a str) -> Option<&'a str>;
    fn error_drop(&self) -> Option<&str>;
    fn payload_import(
        &self,
        module: &str,
        name: &str,
        resolve: &Resolve,
        world: &World,
        ty: &FuncType,
    ) -> Result<Option<Import>>;
    fn module_to_interface(
        &self,
        module: &str,
        resolve: &Resolve,
        items: &IndexMap<WorldKey, WorldItem>,
    ) -> Result<(WorldKey, InterfaceId)>;
    fn strip_post_return<'a>(&self, s: &'a str) -> Option<&'a str>;
    fn match_wit_export<'a>(
        &self,
        export_name: &str,
        resolve: &'a Resolve,
        world: WorldId,
        exports: &'a IndexSet<WorldKey>,
    ) -> Option<(&'a WorldKey, Option<InterfaceId>, &'a Function)>;
    fn match_wit_resource_dtor<'a>(
        &self,
        export_name: &str,
        resolve: &'a Resolve,
        world: WorldId,
        exports: &'a IndexSet<WorldKey>,
    ) -> Option<TypeId>;
}

/// Definition of the "standard" naming scheme which currently starts with
/// "cm32p2". Note that wasm64 is not supported at this time.
struct Standard;

const STANDARD: &'static dyn NameMangling = &Standard;

impl NameMangling for Standard {
    fn import_root(&self) -> &str {
        ""
    }
    fn import_non_root_prefix(&self) -> &str {
        "|"
    }
    fn import_exported_intrinsic_prefix(&self) -> &str {
        "_ex_"
    }
    fn export_memory(&self) -> &str {
        "_memory"
    }
    fn export_initialize(&self) -> &str {
        "_initialize"
    }
    fn export_realloc(&self) -> &str {
        "_realloc"
    }
    fn resource_drop_name<'a>(&self, s: &'a str) -> Option<&'a str> {
        s.strip_suffix("_drop")
    }
    fn resource_new_name<'a>(&self, s: &'a str) -> Option<&'a str> {
        s.strip_suffix("_new")
    }
    fn resource_rep_name<'a>(&self, s: &'a str) -> Option<&'a str> {
        s.strip_suffix("_rep")
    }
    fn task_return_name<'a>(&self, s: &'a str) -> Option<&'a str> {
        _ = s;
        None
    }
    fn task_backpressure(&self) -> Option<&str> {
        None
    }
    fn task_wait(&self) -> Option<&str> {
        None
    }
    fn task_poll(&self) -> Option<&str> {
        None
    }
    fn task_yield(&self) -> Option<&str> {
        None
    }
    fn subtask_drop(&self) -> Option<&str> {
        None
    }
    fn callback_name<'a>(&self, s: &'a str) -> Option<&'a str> {
        _ = s;
        None
    }
    fn async_name<'a>(&self, s: &'a str) -> Option<&'a str> {
        _ = s;
        None
    }
    fn error_drop(&self) -> Option<&str> {
        None
    }
    fn payload_import(
        &self,
        module: &str,
        name: &str,
        resolve: &Resolve,
        world: &World,
        ty: &FuncType,
    ) -> Result<Option<Import>> {
        _ = (module, name, resolve, world, ty);
        Ok(None)
    }
    fn module_to_interface(
        &self,
        interface: &str,
        resolve: &Resolve,
        items: &IndexMap<WorldKey, WorldItem>,
    ) -> Result<(WorldKey, InterfaceId)> {
        for (key, item) in items.iter() {
            let id = match key {
                // Bare keys are matched exactly against `interface`
                WorldKey::Name(name) => match item {
                    WorldItem::Interface { id, .. } if name == interface => *id,
                    _ => continue,
                },
                // ID-identified keys are matched with their "canonical name"
                WorldKey::Interface(id) => {
                    if resolve.canonicalized_id_of(*id).as_deref() != Some(interface) {
                        continue;
                    }
                    *id
                }
            };
            return Ok((key.clone(), id));
        }
        bail!("failed to find world item corresponding to interface `{interface}`")
    }
    fn strip_post_return<'a>(&self, s: &'a str) -> Option<&'a str> {
        s.strip_suffix("_post")
    }
    fn match_wit_export<'a>(
        &self,
        export_name: &str,
        resolve: &'a Resolve,
        world: WorldId,
        exports: &'a IndexSet<WorldKey>,
    ) -> Option<(&'a WorldKey, Option<InterfaceId>, &'a Function)> {
        if let Some(world_export_name) = export_name.strip_prefix("||") {
            let key = exports.get(&WorldKey::Name(world_export_name.to_string()))?;
            match &resolve.worlds[world].exports[key] {
                WorldItem::Function(f) => return Some((key, None, f)),
                _ => return None,
            }
        }

        let (key, id, func_name) =
            self.match_wit_interface(export_name, resolve, world, exports)?;
        let func = resolve.interfaces[id].functions.get(func_name)?;
        Some((key, Some(id), func))
    }

    fn match_wit_resource_dtor<'a>(
        &self,
        export_name: &str,
        resolve: &'a Resolve,
        world: WorldId,
        exports: &'a IndexSet<WorldKey>,
    ) -> Option<TypeId> {
        let (_key, id, name) =
            self.match_wit_interface(export_name.strip_suffix("_dtor")?, resolve, world, exports)?;
        let ty = *resolve.interfaces[id].types.get(name)?;
        match resolve.types[ty].kind {
            TypeDefKind::Resource => Some(ty),
            _ => None,
        }
    }
}

impl Standard {
    fn match_wit_interface<'a, 'b>(
        &self,
        export_name: &'b str,
        resolve: &'a Resolve,
        world: WorldId,
        exports: &'a IndexSet<WorldKey>,
    ) -> Option<(&'a WorldKey, InterfaceId, &'b str)> {
        let world = &resolve.worlds[world];
        let export_name = export_name.strip_prefix("|")?;

        for export in exports {
            let id = match &world.exports[export] {
                WorldItem::Interface { id, .. } => *id,
                WorldItem::Function(_) => continue,
                WorldItem::Type(_) => unreachable!(),
            };
            let remaining = match export {
                WorldKey::Name(name) => export_name.strip_prefix(name),
                WorldKey::Interface(_) => {
                    let prefix = resolve.canonicalized_id_of(id).unwrap();
                    export_name.strip_prefix(&prefix)
                }
            };
            let item_name = match remaining.and_then(|s| s.strip_prefix("|")) {
                Some(name) => name,
                None => continue,
            };
            return Some((export, id, item_name));
        }

        None
    }
}

/// Definition of wit-component's "legacy" naming scheme which predates
/// WebAssembly/component-model#378.
struct Legacy;

const LEGACY: &'static dyn NameMangling = &Legacy;

impl NameMangling for Legacy {
    fn import_root(&self) -> &str {
        "$root"
    }
    fn import_non_root_prefix(&self) -> &str {
        ""
    }
    fn import_exported_intrinsic_prefix(&self) -> &str {
        "[export]"
    }
    fn export_memory(&self) -> &str {
        "memory"
    }
    fn export_initialize(&self) -> &str {
        "_initialize"
    }
    fn export_realloc(&self) -> &str {
        "cabi_realloc"
    }
    fn resource_drop_name<'a>(&self, s: &'a str) -> Option<&'a str> {
        s.strip_prefix("[resource-drop]")
    }
    fn resource_new_name<'a>(&self, s: &'a str) -> Option<&'a str> {
        s.strip_prefix("[resource-new]")
    }
    fn resource_rep_name<'a>(&self, s: &'a str) -> Option<&'a str> {
        s.strip_prefix("[resource-rep]")
    }
    fn task_return_name<'a>(&self, s: &'a str) -> Option<&'a str> {
        s.strip_prefix("[task-return]")
    }
    fn task_backpressure(&self) -> Option<&str> {
        Some("[task-backpressure]")
    }
    fn task_wait(&self) -> Option<&str> {
        Some("[task-wait]")
    }
    fn task_poll(&self) -> Option<&str> {
        Some("[task-poll]")
    }
    fn task_yield(&self) -> Option<&str> {
        Some("[task-yield]")
    }
    fn subtask_drop(&self) -> Option<&str> {
        Some("[subtask-drop]")
    }
    fn callback_name<'a>(&self, s: &'a str) -> Option<&'a str> {
        s.strip_prefix("[callback][async]")
    }
    fn async_name<'a>(&self, s: &'a str) -> Option<&'a str> {
        s.strip_prefix("[async]")
    }
    fn error_drop(&self) -> Option<&str> {
        Some("[error-drop]")
    }
    fn payload_import(
        &self,
        module: &str,
        name: &str,
        resolve: &Resolve,
        world: &World,
        ty: &FuncType,
    ) -> Result<Option<Import>> {
        Ok(
            if let Some((suffix, imported)) = module
                .strip_prefix("[import-payload]")
                .map(|v| (v, true))
                .or_else(|| name.strip_prefix("[export-payload]").map(|v| (v, false)))
            {
                let (key, interface) = if suffix == self.import_root() {
                    (WorldKey::Name(name.to_string()), None)
                } else {
                    let (key, id) = self.module_to_interface(
                        suffix,
                        resolve,
                        if imported {
                            &world.imports
                        } else {
                            &world.exports
                        },
                    )?;
                    (key, Some(id))
                };

                let orig_name = name;

                let (name, async_) = if let Some(name) = self.async_name(name) {
                    (name, true)
                } else {
                    (name, false)
                };

                let info = |payload_key, validate: &dyn Fn() -> Result<()>| {
                    let (function, ty) =
                        get_payload_type(resolve, world, &payload_key, interface, imported)?;
                    validate()?;
                    Ok::<_, anyhow::Error>(PayloadInfo {
                        name: orig_name.to_string(),
                        ty,
                        function,
                        key: key.clone(),
                        interface,
                        imported,
                    })
                };

                // TODO: move
                Some(
                    if let Some(key) = match_payload_prefix(name, "[future-new-") {
                        if async_ {
                            bail!("async `future.new` calls not supported");
                        }
                        Import::FutureNew(info(key, &|| {
                            validate_func_sig(name, &FuncType::new([], [ValType::I32]), ty)
                        })?)
                    } else if let Some(key) = match_payload_prefix(name, "[future-write-") {
                        if !async_ {
                            bail!("TODO: sync `future.write` calls not yet supported");
                        }
                        Import::FutureWrite(info(key, &|| {
                            validate_func_sig(
                                name,
                                &FuncType::new([ValType::I32; 2], [ValType::I32]),
                                ty,
                            )
                        })?)
                    } else if let Some(key) = match_payload_prefix(name, "[future-read-") {
                        if !async_ {
                            bail!("TODO: sync `future.read` calls not yet supported");
                        }
                        Import::FutureRead(info(key, &|| {
                            validate_func_sig(
                                name,
                                &FuncType::new([ValType::I32; 2], [ValType::I32]),
                                ty,
                            )
                        })?)
                    } else if let Some(key) = match_payload_prefix(name, "[future-drop-writer-") {
                        Import::FutureDropWriter(
                            info(key, &|| {
                                validate_func_sig(name, &FuncType::new([ValType::I32], []), ty)
                            })?
                            .ty,
                        )
                    } else if let Some(key) = match_payload_prefix(name, "[future-drop-reader-") {
                        Import::FutureDropReader(
                            info(key, &|| {
                                validate_func_sig(name, &FuncType::new([ValType::I32], []), ty)
                            })?
                            .ty,
                        )
                    } else if let Some(key) = match_payload_prefix(name, "[stream-new-") {
                        if async_ {
                            bail!("async `stream.new` calls not supported");
                        }
                        Import::StreamNew(info(key, &|| {
                            validate_func_sig(name, &FuncType::new([], [ValType::I32]), ty)
                        })?)
                    } else if let Some(key) = match_payload_prefix(name, "[stream-write-") {
                        if !async_ {
                            bail!("TODO: sync `stream.write` calls not yet supported");
                        }
                        Import::StreamWrite(info(key, &|| {
                            validate_func_sig(
                                name,
                                &FuncType::new([ValType::I32; 3], [ValType::I32]),
                                ty,
                            )
                        })?)
                    } else if let Some(key) = match_payload_prefix(name, "[stream-read-") {
                        if !async_ {
                            bail!("TODO: sync `stream.read` calls not yet supported");
                        }
                        Import::StreamRead(info(key, &|| {
                            validate_func_sig(
                                name,
                                &FuncType::new([ValType::I32; 3], [ValType::I32]),
                                ty,
                            )
                        })?)
                    } else if let Some(key) = match_payload_prefix(name, "[stream-drop-writer-") {
                        Import::StreamDropWriter(
                            info(key, &|| {
                                validate_func_sig(name, &FuncType::new([ValType::I32], []), ty)
                            })?
                            .ty,
                        )
                    } else if let Some(key) = match_payload_prefix(name, "[stream-drop-reader-") {
                        Import::StreamDropReader(
                            info(key, &|| {
                                validate_func_sig(name, &FuncType::new([ValType::I32], []), ty)
                            })?
                            .ty,
                        )
                    } else {
                        bail!("unrecognized payload import: {name}");
                    },
                )
            } else {
                None
            },
        )
    }
    fn module_to_interface(
        &self,
        module: &str,
        resolve: &Resolve,
        items: &IndexMap<WorldKey, WorldItem>,
    ) -> Result<(WorldKey, InterfaceId)> {
        // First see if this is a bare name
        let bare_name = WorldKey::Name(module.to_string());
        if let Some(WorldItem::Interface { id, .. }) = items.get(&bare_name) {
            return Ok((bare_name, *id));
        }

        // ... and if this isn't a bare name then it's time to do some parsing
        // related to interfaces, versions, and such. First up the `module` name
        // is parsed as a normal component name from `wasmparser` to see if it's
        // of the "interface kind". If it's not then that means the above match
        // should have been a hit but it wasn't, so an error is returned.
        let kebab_name = ComponentName::new(module, 0);
        let name = match kebab_name.as_ref().map(|k| k.kind()) {
            Ok(ComponentNameKind::Interface(name)) => name,
            _ => bail!("module requires an import interface named `{module}`"),
        };

        // Prioritize an exact match based on versions, so try that first.
        let pkgname = PackageName {
            namespace: name.namespace().to_string(),
            name: name.package().to_string(),
            version: name.version(),
        };
        if let Some(pkg) = resolve.package_names.get(&pkgname) {
            if let Some(id) = resolve.packages[*pkg]
                .interfaces
                .get(name.interface().as_str())
            {
                let key = WorldKey::Interface(*id);
                if items.contains_key(&key) {
                    return Ok((key, *id));
                }
            }
        }

        // If an exact match wasn't found then instead search for the first
        // match based on versions. This means that a core wasm import for
        // "1.2.3" might end up matching an interface at "1.2.4", for example.
        // (or "1.2.2", depending on what's available).
        for (key, _) in items {
            let id = match key {
                WorldKey::Interface(id) => *id,
                WorldKey::Name(_) => continue,
            };
            // Make sure the interface names match
            let interface = &resolve.interfaces[id];
            if interface.name.as_ref().unwrap() != name.interface().as_str() {
                continue;
            }

            // Make sure the package name (without version) matches
            let pkg = &resolve.packages[interface.package.unwrap()];
            if pkg.name.namespace != pkgname.namespace || pkg.name.name != pkgname.name {
                continue;
            }

            let module_version = match name.version() {
                Some(version) => version,
                None => continue,
            };
            let pkg_version = match &pkg.name.version {
                Some(version) => version,
                None => continue,
            };

            // Test if the two semver versions are compatible
            let module_compat = PackageName::version_compat_track(&module_version);
            let pkg_compat = PackageName::version_compat_track(pkg_version);
            if module_compat == pkg_compat {
                return Ok((key.clone(), id));
            }
        }

        bail!("module requires an import interface named `{module}`")
    }
    fn strip_post_return<'a>(&self, s: &'a str) -> Option<&'a str> {
        s.strip_prefix("cabi_post_")
    }
    fn match_wit_export<'a>(
        &self,
        export_name: &str,
        resolve: &'a Resolve,
        world: WorldId,
        exports: &'a IndexSet<WorldKey>,
    ) -> Option<(&'a WorldKey, Option<InterfaceId>, &'a Function)> {
        let world = &resolve.worlds[world];
        for name in exports {
            match &world.exports[name] {
                WorldItem::Function(f) => {
                    if f.legacy_core_export_name(None) == export_name {
                        return Some((name, None, f));
                    }
                }
                WorldItem::Interface { id, .. } => {
                    let string = resolve.name_world_key(name);
                    for (_, func) in resolve.interfaces[*id].functions.iter() {
                        if func.legacy_core_export_name(Some(&string)) == export_name {
                            return Some((name, Some(*id), func));
                        }
                    }
                }

                WorldItem::Type(_) => unreachable!(),
            }
        }

        None
    }

    fn match_wit_resource_dtor<'a>(
        &self,
        export_name: &str,
        resolve: &'a Resolve,
        world: WorldId,
        exports: &'a IndexSet<WorldKey>,
    ) -> Option<TypeId> {
        let world = &resolve.worlds[world];
        for name in exports {
            let id = match &world.exports[name] {
                WorldItem::Interface { id, .. } => *id,
                WorldItem::Function(_) => continue,
                WorldItem::Type(_) => unreachable!(),
            };
            let name = resolve.name_world_key(name);
            let resource = match export_name
                .strip_prefix(&name)
                .and_then(|s| s.strip_prefix("#[dtor]"))
                .and_then(|r| resolve.interfaces[id].types.get(r))
            {
                Some(id) => *id,
                None => continue,
            };

            match resolve.types[resource].kind {
                TypeDefKind::Resource => {}
                _ => continue,
            }

            return Some(resource);
        }

        None
    }
}

pub struct AsyncExportInfo<'a> {
    pub interface: Option<InterfaceId>,
    pub function: &'a Function,
    pub start_import: Option<String>,
    pub return_import: Option<String>,
}

pub struct PayloadInfo<'a> {
    pub interface: Option<InterfaceId>,
    pub function: &'a Function,
    pub ty: TypeId,
    pub future_new: Option<String>,
    pub future_send: Option<String>,
    pub future_receive: Option<String>,
    pub future_drop_sender: Option<String>,
    pub future_drop_receiver: Option<String>,
    pub stream_new: Option<String>,
    pub stream_send: Option<String>,
    pub stream_receive: Option<String>,
    pub stream_drop_sender: Option<String>,
    pub stream_drop_receiver: Option<String>,
}

/// This function validates the following:
///
/// * The `bytes` represent a valid core WebAssembly module.
/// * The module's imports are all satisfied by the given `imports` interfaces
///   or the `adapters` set.
/// * The given default and exported interfaces are satisfied by the module's
///   exports.
///
/// The `ValidatedModule` return value contains the metadata which describes the
/// input module on success. This is then further used to generate a component
/// for this module.
pub fn validate_module<'a>(
    bytes: &'a [u8],
    metadata: &'a Bindgen,
    exports: &IndexSet<WorldKey>,
    adapters: &IndexSet<&str>,
) -> Result<ValidatedModule<'a>> {
    let mut validator = Validator::new();
    let mut types = None;
    let mut import_funcs = IndexMap::new();
    let mut export_funcs = IndexMap::new();
    let mut ret = ValidatedModule {
        required_imports: Default::default(),
        adapters_required: Default::default(),
        needs_error_drop: false,
        needs_task_wait: false,
        has_memory: false,
        realloc: None,
        adapter_realloc: None,
        metadata: &metadata.metadata,
        required_async_funcs: Default::default(),
        required_resource_funcs: Default::default(),
        required_payload_funcs: Default::default(),
        post_returns: Default::default(),
        callbacks: Default::default(),
        initialize: None,
    };

    for payload in Parser::new(0).parse_all(bytes) {
        let payload = payload?;
        if let ValidPayload::End(tys) = validator.payload(&payload)? {
            types = Some(tys);
            break;
        }

        match payload {
            Payload::Version { encoding, .. } if encoding != Encoding::Module => {
                bail!("data is not a WebAssembly module");
            }
            Payload::ImportSection(s) => {
                for import in s {
                    let import = import?;
                    match import.ty {
                        TypeRef::Func(ty) => {
                            let map = match import_funcs.entry(import.module) {
                                Entry::Occupied(e) => e.into_mut(),
                                Entry::Vacant(e) => e.insert(IndexMap::new()),
                            };

                            assert!(map.insert(import.name, ty).is_none());
                        }
                        _ => bail!("module is only allowed to import functions"),
                    }
                }
            }
            Payload::ExportSection(s) => {
                for export in s {
                    let export = export?;

                    match export.kind {
                        ExternalKind::Func => {
                            if is_canonical_function(export.name) {
                                // TODO: validate that the cabi_realloc
                                // function is [i32, i32, i32, i32] -> [i32]
                                if export.name == "cabi_realloc"
                                    || export.name == "canonical_abi_realloc"
                                {
                                    ret.realloc = Some(export.name);
                                }
                                if export.name == "cabi_realloc_adapter" {
                                    ret.adapter_realloc = Some(export.name);
                                }
                            }

                            if export.name == "_initialize" {
                                ret.initialize = Some(export.name);
                            } else {
                                assert!(export_funcs.insert(export.name, export.index).is_none())
                            }
                        }
                        ExternalKind::Memory => {
                            if export.name == "memory" {
                                ret.has_memory = true;
                            }
                        }
                        _ => continue,
                    }
                }
            }
            _ => continue,
        }
    }

    let types = types.unwrap();
    let world = &metadata.resolve.worlds[metadata.world];
    let mut exported_resource_and_async_funcs = Vec::new();
    let mut payload_funcs = Vec::new();

    for (name, funcs) in &import_funcs {
        // An empty module name is indicative of the top-level import namespace,
        // so look for top-level functions here.
        if *name == BARE_FUNC_MODULE_NAME {
            let Imports {
                required,
                needs_error_drop,
                needs_task_wait,
            } = validate_imports_top_level(&metadata.resolve, metadata.world, funcs, &types)?;
            ret.needs_error_drop = needs_error_drop;
            ret.needs_task_wait = needs_task_wait;
            if !(required.funcs.is_empty() && required.resources.is_empty()) {
                let prev = ret.required_imports.insert(BARE_FUNC_MODULE_NAME, required);
                assert!(prev.is_none());
            }
            continue;
        }

        if let Some(interface_name) = name.strip_prefix("[export]") {
            exported_resource_and_async_funcs.push((name, interface_name, &import_funcs[name]));
            continue;
        }

        if let Some((interface_name, imported)) = name
            .strip_prefix("[import-payload]")
            .map(|v| (v, true))
            .or_else(|| name.strip_prefix("[export-payload]").map(|v| (v, false)))
        {
            payload_funcs.push((name, imported, interface_name, &import_funcs[name]));
            continue;
        }

        if adapters.contains(name) {
            let map = ret.adapters_required.entry(name).or_default();
            for (func, ty) in funcs {
                let ty = types[types.core_type_at(*ty).unwrap_sub()].unwrap_func();
                map.insert(func, ty.clone());
            }
        } else {
            match world.imports.get(&world_key(&metadata.resolve, name)) {
                Some(WorldItem::Interface { id: interface, .. }) => {
                    let required = validate_imported_interface(
                        &metadata.resolve,
                        *interface,
                        name,
                        funcs,
                        &types,
                        &mut ret.required_payload_funcs,
                    )
                    .with_context(|| format!("failed to validate import interface `{name}`"))?;
                    let prev = ret.required_imports.insert(name, required);
                    assert!(prev.is_none());
                }
                Some(WorldItem::Function(_) | WorldItem::Type(_)) => {
                    bail!("import `{}` is not an interface", name)
                }
                None => bail!("module requires an import interface named `{name}`"),
            }
        }
    }

    for name in exports {
        validate_exported_item(
            &metadata.resolve,
            &world.exports[name],
            &metadata.resolve.name_world_key(name),
            &export_funcs,
            &types,
            &mut ret.post_returns,
            &mut ret.callbacks,
            &mut ret.required_payload_funcs,
            &mut ret.required_async_funcs,
            &mut ret.required_resource_funcs,
        )?;
    }

    for (name, interface_name, funcs) in exported_resource_and_async_funcs {
        let world_key = world_key(&metadata.resolve, interface_name);
        match world.exports.get(&world_key) {
            Some(WorldItem::Interface { id, .. }) => {
                validate_exported_interface_resource_and_async_imports(
                    &metadata.resolve,
                    *id,
                    name,
                    funcs,
                    &types,
                    &mut ret.required_async_funcs,
                    &mut ret.required_resource_funcs,
                )?;
            }
            _ => bail!("import from `{name}` does not correspond to exported interface"),
        }
    }

    for (name, imported, interface_name, funcs) in payload_funcs {
        let world_key = world_key(&metadata.resolve, interface_name);
        let (item, direction) = if imported {
            (world.imports.get(&world_key), "imported")
        } else {
            (world.exports.get(&world_key), "exported")
        };
        match item {
            Some(WorldItem::Interface { id, .. }) => {
                validate_payload_imports(
                    &metadata.resolve,
                    *id,
                    name,
                    imported,
                    funcs,
                    &types,
                    &mut ret.required_payload_funcs,
                )?;
            }
            _ => bail!("import from `{name}` does not correspond to {direction} interface"),
        }
    }

    Ok(ret)
}

fn validate_exported_interface_resource_and_async_imports<'a, 'b>(
    resolve: &'b Resolve,
    interface: InterfaceId,
    import_module: &str,
    funcs: &IndexMap<&'a str, u32>,
    types: &Types,
    required_async_funcs: &mut IndexMap<String, IndexMap<String, AsyncExportInfo<'b>>>,
    required_resource_funcs: &mut IndexMap<String, IndexMap<String, ResourceInfo>>,
) -> Result<()> {
    let is_resource = |name: &str| match resolve.interfaces[interface].types.get(name) {
        Some(ty) => matches!(resolve.types[*ty].kind, TypeDefKind::Resource),
        None => false,
    };
    let mut async_module = required_async_funcs.get_mut(import_module);
    for (func_name, ty) in funcs {
        if let Some(ref mut info) = async_module {
            if let Some(function_name) = func_name.strip_prefix(ASYNC_START) {
                info[function_name].start_import = Some(func_name.to_string());
                continue;
            }
            if let Some(function_name) = func_name.strip_prefix(ASYNC_RETURN) {
                info[function_name].return_import = Some(func_name.to_string());
                continue;
            }
        }

        if valid_exported_resource_func(func_name, *ty, types, is_resource)?.is_none() {
            bail!("import of `{func_name}` is not a valid resource or async function");
        }
        let info = required_resource_funcs.get_mut(import_module).unwrap();
        if let Some(resource_name) = func_name.strip_prefix(RESOURCE_DROP) {
            info[resource_name].drop_import = Some(func_name.to_string());
            continue;
        }
        if let Some(resource_name) = func_name.strip_prefix(RESOURCE_NEW) {
            info[resource_name].new_import = Some(func_name.to_string());
            continue;
        }
        if let Some(resource_name) = func_name.strip_prefix(RESOURCE_REP) {
            info[resource_name].rep_import = Some(func_name.to_string());
            continue;
        }

        unreachable!();
    }
    Ok(())
}

fn match_payload_prefix(name: &str, prefix: &str) -> Option<(String, usize)> {
    let suffix = name.strip_prefix(prefix)?;
    let index = suffix.find(']')?;
    Some((
        suffix[index + 1..].to_owned(),
        suffix[..index].parse().ok()?,
    ))
}

fn validate_payload_imports<'a, 'b>(
    _resolve: &'b Resolve,
    _interface: InterfaceId,
    module: &str,
    import: bool,
    funcs: &IndexMap<&'a str, u32>,
    _types: &Types,
    required_payload_funcs: &mut IndexMap<
        (String, bool),
        IndexMap<(String, usize), PayloadInfo<'b>>,
    >,
) -> Result<()> {
    // TODO: Verify that the core wasm function signatures match what we expect for each function found below.
    // Presumably any issues will be caught anyway when the final component is validated, but it would be best to
    // catch them early.
    let module = module
        .strip_prefix(if import {
            "[import-payload]"
        } else {
            "[export-payload]"
        })
        .unwrap();
    let info = &mut required_payload_funcs[&(module.to_owned(), import)];
    for (orig_func_name, _ty) in funcs {
        let func_name = orig_func_name
            .strip_prefix("[async]")
            .unwrap_or(orig_func_name);
        if let Some(key) = match_payload_prefix(func_name, "[future-new-") {
            info[&key].future_new = Some(orig_func_name.to_string());
        } else if let Some(key) = match_payload_prefix(func_name, "[future-send-") {
            info[&key].future_send = Some(orig_func_name.to_string());
        } else if let Some(key) = match_payload_prefix(func_name, "[future-receive-") {
            info[&key].future_receive = Some(orig_func_name.to_string());
        } else if let Some(key) = match_payload_prefix(func_name, "[future-drop-sender-") {
            info[&key].future_drop_sender = Some(orig_func_name.to_string());
        } else if let Some(key) = match_payload_prefix(func_name, "[future-drop-receiver-") {
            info[&key].future_drop_receiver = Some(orig_func_name.to_string());
        } else if let Some(key) = match_payload_prefix(func_name, "[stream-new-") {
            info[&key].stream_new = Some(orig_func_name.to_string());
        } else if let Some(key) = match_payload_prefix(func_name, "[stream-send-") {
            info[&key].stream_send = Some(orig_func_name.to_string());
        } else if let Some(key) = match_payload_prefix(func_name, "[stream-receive-") {
            info[&key].stream_receive = Some(orig_func_name.to_string());
        } else if let Some(key) = match_payload_prefix(func_name, "[stream-drop-sender-") {
            info[&key].stream_drop_sender = Some(orig_func_name.to_string());
        } else if let Some(key) = match_payload_prefix(func_name, "[stream-drop-receiver-") {
            info[&key].stream_drop_receiver = Some(orig_func_name.to_string());
        } else {
            bail!("unrecognized payload import: {orig_func_name}");
        }
    }
    Ok(())
}

/// Validation information from an "adapter module" which is distinct from a
/// "main module" validated above.
///
/// This is created by the `validate_adapter_module` function.
pub struct ValidatedAdapter<'a> {
    /// If specified this is the list of required imports from the original set
    /// of possible imports along with the set of functions required from each
    /// imported interface.
    pub required_imports: IndexMap<String, RequiredImports>,

    /// Resource-related functions required and imported which work over
    /// exported resources from the final component.
    ///
    /// Note that this is disjoint from `required_imports` which handles
    /// imported resources and this is only for exported resources. Exported
    /// resources still require intrinsics to be imported into the core module
    /// itself.
    pub required_resource_funcs: IndexMap<String, IndexMap<String, ResourceInfo>>,

    pub required_async_funcs: IndexMap<String, IndexMap<String, AsyncExportInfo<'a>>>,

    pub required_payload_funcs:
        IndexMap<(String, bool), IndexMap<(String, usize), PayloadInfo<'a>>>,

    pub needs_error_drop: bool,
    pub needs_task_wait: bool,

    /// This is the module and field name of the memory import, if one is
    /// specified.
    ///
    /// Due to LLVM codegen this is typically `env::memory` as a totally separate
    /// import from the `required_import` above.
    pub needs_memory: Option<(String, String)>,

    /// Set of names required to be exported from the main module which are
    /// imported by this adapter through the `__main_module__` synthetic export.
    /// This is how the WASI adapter imports `_start`, for example.
    pub needs_core_exports: IndexSet<String>,

    /// Name of the exported function to use for the realloc canonical option
    /// for lowering imports.
    pub import_realloc: Option<String>,

    /// Same as `import_realloc`, but for exported interfaces.
    pub export_realloc: Option<String>,

    /// Metadata about the original adapter module.
    pub metadata: &'a ModuleMetadata,

    /// Post-return functions annotated with `cabi_post_*` in their function
    /// name.
    pub post_returns: IndexSet<String>,

    /// Callback functions annotated with `[callback]*` in their function
    /// name.
    pub callbacks: IndexSet<String>,
}

/// This function will validate the `bytes` provided as a wasm adapter module.
/// Notably this will validate the wasm module itself in addition to ensuring
/// that it has the "shape" of an adapter module. Current constraints are:
///
/// * The adapter module can import only one memory
/// * The adapter module can only import from the name of `interface` specified,
///   and all function imports must match the `required` types which correspond
///   to the lowered types of the functions in `interface`.
///
/// The wasm module passed into this function is the output of the GC pass of an
/// adapter module's original source. This means that the adapter module is
/// already minimized and this is a double-check that the minimization pass
/// didn't accidentally break the wasm module.
///
/// If `is_library` is true, we waive some of the constraints described above,
/// allowing the module to import tables and globals, as well as import
/// functions at the world level, not just at the interface level.
pub fn validate_adapter_module(
    encoder: &ComponentEncoder,
    bytes: &[u8],
    required_by_import: &IndexMap<String, FuncType>,
    exports: &IndexSet<WorldKey>,
    is_library: bool,
    adapters: &IndexSet<&str>,
) -> Result<ValidatedAdapter<'a>> {
    let mut validator = Validator::new();
    let mut import_funcs = IndexMap::new();
    let mut export_funcs = IndexMap::new();
    let mut types = None;
    let mut funcs = Vec::new();
    let mut ret = ValidatedAdapter {
        required_imports: Default::default(),
        required_resource_funcs: Default::default(),
        required_async_funcs: Default::default(),
        required_payload_funcs: Default::default(),
        needs_error_drop: false,
        needs_task_wait: false,
        needs_memory: None,
        needs_core_exports: Default::default(),
        import_realloc: None,
        export_realloc: None,
        metadata,
        post_returns: Default::default(),
        callbacks: Default::default(),
    };

    let mut cabi_realloc = None;
    for payload in Parser::new(0).parse_all(bytes) {
        let payload = payload?;
        match validator.payload(&payload)? {
            ValidPayload::End(tys) => {
                types = Some(tys);
                break;
            }
            ValidPayload::Func(validator, body) => {
                funcs.push((validator, body));
            }
            _ => {}
        }

        match payload {
            Payload::Version { encoding, .. } if encoding != Encoding::Module => {
                bail!("data is not a WebAssembly module");
            }

            Payload::ImportSection(s) => {
                for import in s {
                    let import = import?;
                    match import.ty {
                        TypeRef::Func(ty) => {
                            let map = match import_funcs.entry(import.module) {
                                Entry::Occupied(e) => e.into_mut(),
                                Entry::Vacant(e) => e.insert(IndexMap::new()),
                            };

                            assert!(map.insert(import.name, ty).is_none());
                        }

                        // A memory is allowed to be imported into the adapter
                        // module so that's skipped here
                        TypeRef::Memory(_) => {
                            ret.needs_memory =
                                Some((import.module.to_string(), import.name.to_string()));
                        }

                        TypeRef::Global(_) | TypeRef::Table(_) if is_library => (),

                        _ => {
                            bail!("adapter module is only allowed to import functions and memories")
                        }
                    }
                }
            }
            Payload::ExportSection(s) => {
                for export in s {
                    let export = export?;

                    match export.kind {
                        ExternalKind::Func => {
                            export_funcs.insert(export.name, export.index);
                            if export.name == "cabi_export_realloc" {
                                ret.export_realloc = Some(export.name.to_string());
                            }
                            if export.name == "cabi_import_realloc" {
                                ret.import_realloc = Some(export.name.to_string());
                            }
                            if export.name == "cabi_realloc" {
                                cabi_realloc = Some(export.name.to_string());
                            }
                        }
                        _ => continue,
                    }
                }
            }
            _ => continue,
        }
    }

    if is_library {
        // If we're looking at a library, it may only export the
        // `wit-bindgen`-generated `cabi_realloc` rather than the
        // `cabi_import_realloc` and `cabi_export_realloc` functions, so we'll
        // use whatever's available.
        ret.export_realloc = ret.export_realloc.or_else(|| cabi_realloc.clone());
        ret.import_realloc = ret.import_realloc.or_else(|| cabi_realloc);
    }

    let mut resources = Default::default();
    for (validator, body) in funcs {
        let mut validator = validator.into_validator(resources);
        validator.validate(&body)?;
        resources = validator.into_allocations();
    }

    let types = types.unwrap();
    let mut exported_resource_and_async_funcs = Vec::new();
    let mut payload_funcs = Vec::new();

    for (name, funcs) in &import_funcs {
        if *name == MAIN_MODULE_IMPORT_NAME {
            ret.needs_core_exports
                .extend(funcs.iter().map(|(name, _ty)| name.to_string()));
            continue;
        }

        // An empty module name is indicative of the top-level import namespace,
        // so look for top-level functions here.
        if *name == BARE_FUNC_MODULE_NAME {
            let Imports {
                required,
                needs_error_drop,
                needs_task_wait,
            } = validate_imports_top_level(resolve, world, funcs, &types)?;
            ret.needs_error_drop = needs_error_drop;
            ret.needs_task_wait = needs_task_wait;
            if !(required.funcs.is_empty() && required.resources.is_empty()) {
                let prev = ret
                    .required_imports
                    .insert(BARE_FUNC_MODULE_NAME.to_string(), required);
                assert!(prev.is_none());
            }
            continue;
        }

        if let Some(interface_name) = name.strip_prefix("[export]") {
            exported_resource_and_async_funcs.push((name, interface_name, &import_funcs[name]));
            continue;
        }

        if let Some((interface_name, imported)) = name
            .strip_prefix("[import-payload]")
            .map(|v| (v, true))
            .or_else(|| name.strip_prefix("[export-payload]").map(|v| (v, false)))
        {
            payload_funcs.push((name, imported, interface_name, &import_funcs[name]));
            continue;
        }

        if !(is_library && adapters.contains(name)) {
            match resolve.worlds[world].imports.get(&world_key(resolve, name)) {
                Some(WorldItem::Interface { id: interface, .. }) => {
                    let required = validate_imported_interface(
                        resolve,
                        *interface,
                        name,
                        funcs,
                        &types,
                        &mut ret.required_payload_funcs,
                    )
                    .with_context(|| format!("failed to validate import interface `{name}`"))?;
                    let prev = ret.required_imports.insert(name.to_string(), required);
                    assert!(prev.is_none());
                }
                None | Some(WorldItem::Function(_) | WorldItem::Type(_)) => {
                    if !is_library {
                        bail!(
                            "adapter module requires an import interface named `{}`",
                            name
                        )
                    }
                }
            }
        }
    }

    if let Some(required) = required_by_import {
        for (name, ty) in required {
            let idx = match export_funcs.get(name) {
                Some(idx) => *idx,
                None => bail!("adapter module did not export `{name}`"),
            };
            let id = types.core_function_at(idx);
            let actual = types[id].unwrap_func();
            validate_func_sig(name, ty, actual)?;
        }
    }

    let world = &resolve.worlds[world];

    for name in exports {
        validate_exported_item(
            resolve,
            &world.exports[name],
            &resolve.name_world_key(name),
            &export_funcs,
            &types,
            &mut ret.post_returns,
            &mut ret.callbacks,
            &mut ret.required_payload_funcs,
            &mut ret.required_async_funcs,
            &mut ret.required_resource_funcs,
        )?;
    }

    for (name, interface_name, funcs) in exported_resource_and_async_funcs {
        let world_key = world_key(resolve, interface_name);
        match world.exports.get(&world_key) {
            Some(WorldItem::Interface { id, .. }) => {
                validate_exported_interface_resource_and_async_imports(
                    resolve,
                    *id,
                    name,
                    funcs,
                    &types,
                    &mut ret.required_async_funcs,
                    &mut ret.required_resource_funcs,
                )?;
            }
            _ => bail!("import from `{name}` does not correspond to exported interface"),
        }
    }

    for (name, imported, interface_name, funcs) in payload_funcs {
        let world_key = world_key(resolve, interface_name);
        let (item, direction) = if imported {
            (world.imports.get(&world_key), "imported")
        } else {
            (world.exports.get(&world_key), "exported")
        };
        match item {
            Some(WorldItem::Interface { id, .. }) => {
                validate_payload_imports(
                    resolve,
                    *id,
                    name,
                    imported,
                    funcs,
                    &types,
                    &mut ret.required_payload_funcs,
                )?;
            }
            _ => bail!("import from `{name}` does not correspond to {direction} interface"),
        }
    }

    Ok(ret)
}

fn world_key(resolve: &Resolve, name: &str) -> WorldKey {
    let kebab_name = ComponentName::new(name, 0);
    let (pkgname, interface) = match kebab_name.as_ref().map(|k| k.kind()) {
        Ok(ComponentNameKind::Interface(name)) => {
            let pkgname = PackageName {
                namespace: name.namespace().to_string(),
                name: name.package().to_string(),
                version: name.version(),
            };
            (pkgname, name.interface().as_str())
        }
        _ => return WorldKey::Name(name.to_string()),
    };
    match resolve
        .package_names
        .get(&pkgname)
        .and_then(|p| resolve.packages[*p].interfaces.get(interface))
    {
        Some(id) => WorldKey::Interface(*id),
        None => WorldKey::Name(name.to_string()),
    }
}

struct Imports {
    required: RequiredImports,
    needs_error_drop: bool,
    needs_task_wait: bool,
}

fn validate_imports_top_level(
    resolve: &Resolve,
    world: WorldId,
    funcs: &IndexMap<&str, u32>,
    types: &Types,
) -> Result<Imports> {
    // TODO: handle top-level required async, future, and stream built-in imports here
    let is_resource = |name: &str| match resolve.worlds[world]
        .imports
        .get(&WorldKey::Name(name.to_string()))
    {
        Some(WorldItem::Type(r)) => {
            matches!(resolve.types[*r].kind, TypeDefKind::Resource)
        }
        _ => false,
    };
    let mut imports = Imports {
        required: RequiredImports::default(),
        needs_error_drop: false,
        needs_task_wait: false,
    };
    for (name, ty) in funcs {
        {
            if *name == "[error-drop]" {
                imports.needs_error_drop = true;
                continue;
            }

            if *name == "[task-wait]" {
                imports.needs_task_wait = true;
                continue;
            }

            let (abi, name) = if let Some(name) = name.strip_prefix("[async]") {
                (AbiVariant::GuestImportAsync, name)
            } else {
                (AbiVariant::GuestImport, *name)
            };
            match resolve.worlds[world].imports.get(&world_key(resolve, name)) {
                Some(WorldItem::Function(func)) => {
                    let ty = types[types.core_type_at(*ty).unwrap_sub()].unwrap_func();
                    validate_func(resolve, ty, &func, abi)?;
                }
                Some(_) => bail!("expected world top-level import `{name}` to be a function"),
                None => match valid_imported_resource_func(name, *ty, types, is_resource)? {
                    Some(name) => {
                        imports.required.resources.insert(name.to_string());
                    }
                    None => bail!("no top-level imported function `{name}` specified"),
                },
            }
        }
        imports.required.funcs.insert(name.to_string());
    }
    Ok(imports)
}

fn valid_imported_resource_func<'a>(
    func_name: &'a str,
    ty: u32,
    types: &Types,
    is_resource: impl Fn(&str) -> bool,
) -> Result<Option<&'a str>> {
    if let Some(resource_name) = func_name.strip_prefix(RESOURCE_DROP) {
        if is_resource(resource_name) {
            let ty = types[types.core_type_at(ty).unwrap_sub()].unwrap_func();
            let expected = FuncType::new([ValType::I32], []);
            validate_func_sig(func_name, &expected, ty)?;
            return Ok(Some(resource_name));
        }
    }
    Ok(None)
}

fn valid_exported_resource_func<'a>(
    func_name: &'a str,
    ty: u32,
    types: &Types,
    is_resource: impl Fn(&str) -> bool,
) -> Result<Option<&'a str>> {
    if let Some(name) = valid_imported_resource_func(func_name, ty, types, &is_resource)? {
        return Ok(Some(name));
    }
    if let Some(resource_name) = func_name
        .strip_prefix(RESOURCE_REP)
        .or_else(|| func_name.strip_prefix(RESOURCE_NEW))
    {
        if is_resource(resource_name) {
            let ty = types[types.core_type_at(ty).unwrap_sub()].unwrap_func();
            let expected = FuncType::new([ValType::I32], [ValType::I32]);
            validate_func_sig(func_name, &expected, ty)?;
            return Ok(Some(resource_name));
        }
    }
    Ok(None)
}

fn find_payloads<'a>(
    resolve: &'a Resolve,
    interface: Option<InterfaceId>,
    function: &'a Function,
    payload_map: &mut IndexMap<(String, usize), PayloadInfo<'a>>,
) {
    let types = function.find_futures_and_streams(resolve);
    for (index, ty) in types.iter().enumerate() {
        payload_map.insert(
            (function.name.clone(), index),
            PayloadInfo {
                interface,
                function,
                ty: *ty,
                future_new: None,
                future_send: None,
                future_receive: None,
                future_drop_sender: None,
                future_drop_receiver: None,
                stream_new: None,
                stream_send: None,
                stream_receive: None,
                stream_drop_sender: None,
                stream_drop_receiver: None,
            },
        );
    }
}

fn validate_imported_interface<'a>(
    resolve: &'a Resolve,
    interface: InterfaceId,
    name: &str,
    imports: &IndexMap<&str, u32>,
    types: &Types,
    required_payload_funcs: &mut IndexMap<
        (String, bool),
        IndexMap<(String, usize), PayloadInfo<'a>>,
    >,
) -> Result<RequiredImports> {
    let mut required = RequiredImports::default();
    let is_resource = |name: &str| {
        let ty = match resolve.interfaces[interface].types.get(name) {
            Some(ty) => *ty,
            None => return None,
        };
        matches!(resolve.types[ty].kind, TypeDefKind::Resource)
    };
    for (func_name, ty) in imports {
        {
            let (abi, func_name) = if let Some(name) = func_name.strip_prefix("[async]") {
                (AbiVariant::GuestImportAsync, name)
            } else {
                (AbiVariant::GuestImport, *func_name)
            };
            match resolve.interfaces[interface].functions.get(func_name) {
                Some(f) => {
                    let ty = types[types.core_type_at(*ty).unwrap_sub()].unwrap_func();
                    validate_func(resolve, ty, f, abi)?;
                    find_payloads(
                        resolve,
                        Some(interface),
                        f,
                        required_payload_funcs
                            .entry((name.to_string(), true))
                            .or_default(),
                    );
                }
                None => match valid_imported_resource_func(func_name, *ty, types, is_resource)? {
                    Some(name) => {
                        required.resources.insert(name.to_string());
                    }
                    None => bail!(
                        "import interface `{name}` is missing function \
                         `{func_name}` that is required by the module",
                    ),
                },
            }
        }
    }
}

fn resource_test_for_world<'a>(
    resolve: &'a Resolve,
    id: WorldId,
) -> impl Fn(&str) -> Option<TypeId> + 'a {
    let world = &resolve.worlds[id];
    move |name: &str| match world.imports.get(&WorldKey::Name(name.to_string()))? {
        WorldItem::Type(r) => {
            if matches!(resolve.types[*r].kind, TypeDefKind::Resource) {
                Some(*r)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn validate_func(
    resolve: &Resolve,
    ty: &wasmparser::FuncType,
    func: &Function,
    abi: AbiVariant,
) -> Result<()> {
    validate_func_sig(
        &func.name,
        &wasm_sig_to_func_type(resolve.wasm_signature(abi, func)),
        ty,
    )
}

fn validate_post_return(
    resolve: &Resolve,
    ty: &wasmparser::FuncType,
    func: &Function,
) -> Result<()> {
    // The expected signature of a post-return function is to take all the
    // parameters that are returned by the guest function and then return no
    // results. Model this by calculating the signature of `func` and then
    // moving its results into the parameters list while emptying out the
    // results.
    let mut sig = resolve.wasm_signature(AbiVariant::GuestExport, func);
    sig.params = mem::take(&mut sig.results);
    validate_func_sig(
        &format!("{} post-return", func.name),
        &wasm_sig_to_func_type(sig),
        ty,
    )
}

fn validate_callback(ty: &FuncType, func: &Function) -> Result<()> {
    validate_func_sig(
        &format!("{} callback", func.name),
        &FuncType::new([ValType::I32; 4], [ValType::I32]),
        ty,
    )
}

fn validate_func_sig(name: &str, expected: &FuncType, ty: &wasmparser::FuncType) -> Result<()> {
    if ty != expected {
        bail!(
            "type mismatch for function `{}`: expected `{:?} -> {:?}` but found `{:?} -> {:?}`",
            name,
            expected.params(),
            expected.results(),
            ty.params(),
            ty.results()
        );
    }

    Ok(())
}

fn match_payload_prefix(name: &str, prefix: &str) -> Option<(String, usize)> {
    let suffix = name.strip_prefix(prefix)?;
    let index = suffix.find(']')?;
    Some((
        suffix[index + 1..].to_owned(),
        suffix[..index].parse().ok()?,
    ))
}

fn get_payload_type(
    resolve: &Resolve,
    world: &World,
    (name, index): &(String, usize),
    interface: Option<InterfaceId>,
    imported: bool,
) -> Result<(Function, TypeId)> {
    let function = get_function(resolve, world, name, interface, imported)?;
    let ty = function.find_futures_and_streams(resolve)[*index];
    Ok((function, ty))
}

fn get_function(
    resolve: &Resolve,
    world: &World,
    name: &str,
    interface: Option<InterfaceId>,
    imported: bool,
) -> Result<Function> {
    let function = if let Some(id) = interface {
        resolve.interfaces[id]
            .functions
            .get(name)
            .cloned()
            .map(WorldItem::Function)
    } else if imported {
        world
            .imports
            .get(&WorldKey::Name(name.to_string()))
            .cloned()
    } else {
        world
            .exports
            .get(&WorldKey::Name(name.to_string()))
            .cloned()
    };
    let Some(WorldItem::Function(function)) = function else {
        bail!("no export `{name}` export found");
    };
    Ok(function)
}

fn validate_task_return(resolve: &Resolve, ty: &FuncType, function: &Function) -> Result<()> {
    // TODO
    _ = (resolve, ty, function);
    Ok(())
}
