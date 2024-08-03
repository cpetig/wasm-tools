use std::{
    num::NonZeroUsize,
    ops::{Add, AddAssign},
};

use crate::{FlagsRepr, Int, Resolve, Type, TypeDef, TypeDefKind};

/// Architecture specific alignment
#[derive(Eq, PartialEq, PartialOrd, Clone, Copy, Debug)]
pub enum Alignment {
    Pointer,
    Bytes(NonZeroUsize),
}

impl Default for Alignment {
    fn default() -> Self {
        Alignment::Bytes(NonZeroUsize::new(1).unwrap())
    }
}

impl std::fmt::Display for Alignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Alignment::Pointer => f.write_str("ptr"),
            Alignment::Bytes(b) => f.write_fmt(format_args!("{}", b.get())),
        }
    }
}

impl Ord for Alignment {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Alignment::Pointer, Alignment::Pointer) => std::cmp::Ordering::Equal,
            (Alignment::Pointer, Alignment::Bytes(b)) => {
                if b.get() > 4 {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            }
            (Alignment::Bytes(b), Alignment::Pointer) => {
                if b.get() > 4 {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            }
            (Alignment::Bytes(a), Alignment::Bytes(b)) => a.cmp(b),
        }
    }
}

impl Alignment {
    /// for easy migration this gives you the value for wasm32
    pub fn align_wasm32(&self) -> usize {
        match self {
            Alignment::Pointer => 4,
            Alignment::Bytes(bytes) => bytes.get(),
        }
    }

    pub fn format(&self, ptrsize_expr: &str) -> String {
        match self {
            Alignment::Pointer => ptrsize_expr.into(),
            Alignment::Bytes(bytes) => format!("{}", bytes.get()),
        }
    }
}

/// Architecture specific measurement of position,
/// the combined amount in bytes is
/// `bytes + if 4 < core::mem::size_of::<*const u8>() { add_for_64bit } else { 0 }`
#[derive(Default, Clone, Copy, Eq, PartialEq, Debug)]
pub struct ArchitectureSize {
    /// exact value for 32-bit pointers
    pub bytes: usize,
    /// amount of bytes to add for 64-bit architecture
    pub add_for_64bit: usize,
}

impl Add<ArchitectureSize> for ArchitectureSize {
    type Output = ArchitectureSize;

    fn add(self, rhs: ArchitectureSize) -> Self::Output {
        ArchitectureSize::new(
            self.bytes + rhs.bytes,
            self.add_for_64bit + rhs.add_for_64bit,
        )
    }
}

impl AddAssign<ArchitectureSize> for ArchitectureSize {
    fn add_assign(&mut self, rhs: ArchitectureSize) {
        self.bytes += rhs.bytes;
        self.add_for_64bit += rhs.add_for_64bit;
    }
}

impl From<Alignment> for ArchitectureSize {
    fn from(align: Alignment) -> Self {
        match align {
            Alignment::Bytes(bytes) => ArchitectureSize::new(bytes.get(), 0),
            Alignment::Pointer => ArchitectureSize::new(4, 4),
        }
    }
}

impl std::fmt::Display for ArchitectureSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.format("ptrsz"))
    }
}

impl ArchitectureSize {
    pub fn new(bytes: usize, add_for_64bit: usize) -> Self {
        Self {
            bytes,
            add_for_64bit,
        }
    }

    pub fn max<B: std::borrow::Borrow<Self>>(&self, other: B) -> Self {
        let new_bytes = self.bytes.max(other.borrow().bytes);
        Self::new(
            new_bytes,
            (self.bytes + self.add_for_64bit)
                .max(other.borrow().bytes + other.borrow().add_for_64bit)
                - new_bytes,
        )
    }

    pub fn add_bytes(&self, b: usize) -> Self {
        Self::new(self.bytes + b, self.add_for_64bit)
    }

    /// The effective offset/size is
    /// `constant_bytes() + core::mem::size_of::<*const u8>() * pointers_to_add()`
    pub fn constant_bytes(&self) -> usize {
        self.bytes - self.add_for_64bit
    }

    pub fn pointers_to_add(&self) -> usize {
        self.add_for_64bit / 4
    }

    /// Shortcut for compatibility with previous versions
    pub fn size_wasm32(&self) -> usize {
        self.bytes
    }

    /// prefer this over >0
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    // create a suitable expression in bytes from a pointer size argument
    pub fn format(&self, ptrsize_expr: &str) -> String {
        if self.add_for_64bit != 0 {
            if self.bytes > self.add_for_64bit {
                // both
                format!(
                    "({}+{}*{ptrsize_expr})",
                    self.constant_bytes(),
                    self.pointers_to_add()
                )
            } else if self.add_for_64bit == 4 {
                // one pointer
                ptrsize_expr.into()
            } else {
                // only pointer
                format!("({}*{ptrsize_expr})", self.pointers_to_add())
            }
        } else {
            // only bytes
            format!("{}", self.constant_bytes())
        }
    }
}

/// Information per structure element
#[derive(Default)]
pub struct ElementInfo {
    pub size: ArchitectureSize,
    pub align: Alignment,
}

impl From<Alignment> for ElementInfo {
    fn from(align: Alignment) -> Self {
        ElementInfo {
            size: align.into(),
            align,
        }
    }
}

impl ElementInfo {
    fn new(size: ArchitectureSize, align: Alignment) -> Self {
        Self { size, align }
    }
}

/// Collect size and alignment for sub-elements of a structure
#[derive(Default)]
pub struct SizeAlign64 {
    map: Vec<ElementInfo>,
    symmetric: bool,
}

impl SizeAlign64 {
    // pub fn new() -> Self {
    //     Default::default()
    // }

    pub fn new_symmetric() -> Self {
        Self {
            map: Vec::new(),
            symmetric: true,
        }
    }

    pub fn fill(&mut self, resolve: &Resolve) {
        self.map = Vec::new();
        for (_, ty) in resolve.types.iter() {
            let pair = self.calculate(ty);
            self.map.push(pair);
        }
    }

    fn calculate(&self, ty: &TypeDef) -> ElementInfo {
        match &ty.kind {
            TypeDefKind::Type(t) => ElementInfo::new(self.size(t), self.align(t)),
            TypeDefKind::List(_) => {
                ElementInfo::new(ArchitectureSize::new(8, 8), Alignment::Pointer)
            }
            TypeDefKind::Record(r) => self.record(r.fields.iter().map(|f| &f.ty)),
            TypeDefKind::Tuple(t) => self.record(t.types.iter()),
            TypeDefKind::Flags(f) => match f.repr() {
                FlagsRepr::U8 => int_size_align(Int::U8),
                FlagsRepr::U16 => int_size_align(Int::U16),
                FlagsRepr::U32(n) => ElementInfo::new(
                    ArchitectureSize::new(n * 4, 0),
                    Alignment::Bytes(NonZeroUsize::new(4).unwrap()),
                ),
            },
            TypeDefKind::Variant(v) => self.variant(v.tag(), v.cases.iter().map(|c| c.ty.as_ref())),
            TypeDefKind::Enum(e) => self.variant(e.tag(), []),
            TypeDefKind::Option(t) => self.variant(Int::U8, [Some(t)]),
            TypeDefKind::Result(r) => self.variant(Int::U8, [r.ok.as_ref(), r.err.as_ref()]),
            // A resource is represented as an index.
            // A future is represented as an index.
            // A stream is represented as an index.
            TypeDefKind::Handle(_) | TypeDefKind::Future(_) | TypeDefKind::Stream(_) => {
                if self.symmetric {
                    ElementInfo::new(ArchitectureSize::new(4, 4), Alignment::Pointer)
                } else {
                    int_size_align(Int::U32)
                }
            }
            // An error is represented as an index.
            TypeDefKind::Error => int_size_align(Int::U32),
            // This shouldn't be used for anything since raw resources aren't part of the ABI -- just handles to
            // them.
            TypeDefKind::Resource => ElementInfo::new(
                ArchitectureSize::new(usize::MAX, usize::MAX),
                Alignment::Bytes(NonZeroUsize::new(usize::MAX).unwrap()),
            ),
            TypeDefKind::Unknown => unreachable!(),
        }
    }

    pub fn size(&self, ty: &Type) -> ArchitectureSize {
        match ty {
            Type::Bool | Type::U8 | Type::S8 => ArchitectureSize::new(1, 0),
            Type::U16 | Type::S16 => ArchitectureSize::new(2, 0),
            Type::U32 | Type::S32 | Type::F32 | Type::Char => ArchitectureSize::new(4, 0),
            Type::U64 | Type::S64 | Type::F64 => ArchitectureSize::new(8, 0),
            Type::String => ArchitectureSize::new(8, 8),
            Type::Id(id) => self.map[id.index()].size,
        }
    }

    pub fn align(&self, ty: &Type) -> Alignment {
        match ty {
            Type::Bool | Type::U8 | Type::S8 => Alignment::Bytes(NonZeroUsize::new(1).unwrap()),
            Type::U16 | Type::S16 => Alignment::Bytes(NonZeroUsize::new(2).unwrap()),
            Type::U32 | Type::S32 | Type::F32 | Type::Char => {
                Alignment::Bytes(NonZeroUsize::new(4).unwrap())
            }
            Type::U64 | Type::S64 | Type::F64 => Alignment::Bytes(NonZeroUsize::new(8).unwrap()),
            Type::String => Alignment::Pointer,
            Type::Id(id) => self.map[id.index()].align,
        }
    }

    pub fn field_offsets<'a>(
        &self,
        types: impl IntoIterator<Item = &'a Type>,
    ) -> Vec<(ArchitectureSize, &'a Type)> {
        let mut cur = ArchitectureSize::default();
        types
            .into_iter()
            .map(|ty| {
                let ret = align_to_arch(cur, self.align(ty));
                cur = ret + self.size(ty);
                (ret, ty)
            })
            .collect()
    }

    pub fn payload_offset<'a>(
        &self,
        tag: Int,
        cases: impl IntoIterator<Item = Option<&'a Type>>,
    ) -> ArchitectureSize {
        let mut max_align = Alignment::default();
        for ty in cases {
            if let Some(ty) = ty {
                max_align = max_align.max(self.align(ty));
            }
        }
        let tag_size = int_size_align(tag).size;
        align_to_arch(tag_size, max_align)
    }

    pub fn record<'a>(&self, types: impl Iterator<Item = &'a Type>) -> ElementInfo {
        let mut size = ArchitectureSize::default();
        let mut align = Alignment::default();
        for ty in types {
            let field_size = self.size(ty);
            let field_align = self.align(ty);
            size = align_to_arch(size, field_align) + field_size;
            align = align.max(field_align);
        }
        ElementInfo::new(align_to_arch(size, align), align)
    }

    pub fn params<'a>(&self, types: impl IntoIterator<Item = &'a Type>) -> ElementInfo {
        self.record(types.into_iter())
    }

    fn variant<'a>(
        &self,
        tag: Int,
        types: impl IntoIterator<Item = Option<&'a Type>>,
    ) -> ElementInfo {
        let ElementInfo {
            size: discrim_size,
            align: discrim_align,
        } = int_size_align(tag);
        let mut case_size = ArchitectureSize::default();
        let mut case_align = Alignment::default();
        for ty in types {
            if let Some(ty) = ty {
                case_size = case_size.max(&self.size(ty));
                case_align = case_align.max(self.align(ty));
            }
        }
        let align = discrim_align.max(case_align);
        ElementInfo::new(
            align_to_arch(align_to_arch(discrim_size, case_align) + case_size, align),
            align,
        )
    }
}

fn int_size_align(i: Int) -> ElementInfo {
    match i {
        Int::U8 => Alignment::Bytes(NonZeroUsize::new(1).unwrap()),
        Int::U16 => Alignment::Bytes(NonZeroUsize::new(2).unwrap()),
        Int::U32 => Alignment::Bytes(NonZeroUsize::new(4).unwrap()),
        Int::U64 => Alignment::Bytes(NonZeroUsize::new(8).unwrap()),
    }
    .into()
}

/// Increase `val` to a multiple of `align`;
/// `align` must be a power of two
pub(crate) fn align_to(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

/// Increase `val` to a multiple of `align`, with special handling for pointers;
/// `align` must be a power of two or `Alignment::Pointer`
pub fn align_to_arch(val: ArchitectureSize, align: Alignment) -> ArchitectureSize {
    match align {
        Alignment::Pointer => {
            let new_bytes = align_to(val.bytes, 4);
            let unaligned64 = new_bytes + val.add_for_64bit;
            ArchitectureSize::new(
                new_bytes,
                // increase if necessary for 64bit alignment
                val.add_for_64bit
                    + if unaligned64 != align_to(unaligned64, 8) {
                        4
                    } else {
                        0
                    },
            )
        }
        Alignment::Bytes(align_bytes) => {
            let new_bytes = align_to(val.bytes, align_bytes.get());
            ArchitectureSize::new(
                new_bytes,
                align_to(val.bytes + val.add_for_64bit, align_bytes.get()) - new_bytes,
            )
        }
    }
}

/// Compatibility with older versions:
/// Collect size and alignment for sub-elements of a structure, simpler interface only supporting wasm32
#[derive(Default)]
pub struct SizeAlign32(SizeAlign64);

impl SizeAlign32 {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn fill(&mut self, resolve: &Resolve) {
        self.0.fill(resolve);
    }
    pub fn size(&self, ty: &Type) -> usize {
        self.0.size(ty).size_wasm32()
    }
    pub fn align(&self, ty: &Type) -> usize {
        self.0.align(ty).align_wasm32()
    }
    pub fn field_offsets<'a>(
        &self,
        types: impl IntoIterator<Item = &'a Type>,
    ) -> Vec<(usize, &'a Type)> {
        self.0
            .field_offsets(types)
            .drain(..)
            .map(|(offs, ty)| (offs.size_wasm32(), ty))
            .collect()
    }
    pub fn payload_offset<'a>(
        &self,
        tag: Int,
        cases: impl IntoIterator<Item = Option<&'a Type>>,
    ) -> usize {
        self.0.payload_offset(tag, cases).size_wasm32()
    }
    pub fn record<'a>(&self, types: impl Iterator<Item = &'a Type>) -> (usize, usize) {
        let info = self.0.record(types);
        (info.size.size_wasm32(), info.align.align_wasm32())
    }
    pub fn params<'a>(&self, types: impl IntoIterator<Item = &'a Type>) -> (usize, usize) {
        self.record(types.into_iter())
    }
}

/// compatibility alias
pub type SizeAlign = SizeAlign32;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn align() {
        // u8 + ptr
        assert_eq!(
            align_to_arch(ArchitectureSize::new(1, 0), Alignment::Pointer),
            ArchitectureSize::new(4, 4)
        );
        // u8 + u64
        assert_eq!(
            align_to_arch(
                ArchitectureSize::new(1, 0),
                Alignment::Bytes(NonZeroUsize::new(8).unwrap())
            ),
            ArchitectureSize::new(8, 0)
        );
        // u8 + u32
        assert_eq!(
            align_to_arch(
                ArchitectureSize::new(1, 0),
                Alignment::Bytes(NonZeroUsize::new(4).unwrap())
            ),
            ArchitectureSize::new(4, 0)
        );
        // ptr + u64
        assert_eq!(
            align_to_arch(
                ArchitectureSize::new(4, 4),
                Alignment::Bytes(NonZeroUsize::new(8).unwrap())
            ),
            ArchitectureSize::new(8, 0)
        );
        // u32 + ptr
        assert_eq!(
            align_to_arch(ArchitectureSize::new(4, 0), Alignment::Pointer),
            ArchitectureSize::new(4, 4)
        );
        // u32, ptr + u64
        assert_eq!(
            align_to_arch(
                ArchitectureSize::new(8, 8),
                Alignment::Bytes(NonZeroUsize::new(8).unwrap())
            ),
            ArchitectureSize::new(8, 8)
        );
        // ptr, u8 + u64
        assert_eq!(
            align_to_arch(
                ArchitectureSize::new(5, 4),
                Alignment::Bytes(NonZeroUsize::new(8).unwrap())
            ),
            ArchitectureSize::new(8, 8)
        );
        // ptr, u8 + ptr
        assert_eq!(
            align_to_arch(ArchitectureSize::new(5, 4), Alignment::Pointer),
            ArchitectureSize::new(8, 8)
        );

        assert_eq!(
            ArchitectureSize::new(12, 0).max(&ArchitectureSize::new(8, 8)),
            ArchitectureSize::new(12, 4)
        );
    }
}
