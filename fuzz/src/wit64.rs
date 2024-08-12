use arbitrary::{Result, Unstructured};
use std::path::Path;
use wit_parser::{AddressSize, SizeAlign};

pub fn run(u: &mut Unstructured<'_>) -> Result<()> {
    let wasm = u.arbitrary().and_then(|config| {
        log::debug!("config: {config:#?}");
        wit_smith::smith(&config, u)
    })?;
    write_file("doc.wasm", &wasm);
    let r1 = wit_component::decode(&wasm).unwrap();
    let r1 = r1.resolve();
    let r2 = wit_component2::decode(&wasm).unwrap();
    let r2 = r2.resolve();

    let mut s32 = SizeAlign::new(AddressSize::Wasm32);
    let mut s64 = SizeAlign::new(AddressSize::Wasm64);

    s32.fill(r1);
    s64.fill(r1);

    let mut alt = wit_parser2::SizeAlign64::default();
    alt.fill(r2);

    for ((t1, _), (t2, _)) in r1.types.iter().zip(r2.types.iter()) {
        let t1 = &wit_parser::Type::Id(t1);
        let t2 = &wit_parser2::Type::Id(t2);
        let (s32, a32) = (s32.size(t1), s32.align(t1));
        let (s64, a64) = (s64.size(t1), s64.align(t1));
        let (salt, aalt) = (alt.size(t2), alt.align(t2));

        assert!(s32 <= s64);
        assert!(a32 <= a64);

        assert_eq!(a32, aalt.align_wasm32());
        assert_eq!(
            a64,
            match aalt {
                wit_parser2::Alignment::Bytes(b) => b.get(),
                wit_parser2::Alignment::Pointer => 8,
            }
        );

        assert_eq!(s32, salt.size_wasm32());
        assert_eq!(s64, salt.bytes + salt.add_for_64bit);
    }

    Ok(())
}

fn write_file(path: &str, contents: impl AsRef<[u8]>) {
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }
    log::debug!("writing file {path}");
    let contents = contents.as_ref();
    let path = Path::new(path);
    std::fs::write(path, contents).unwrap();
    if path.extension().and_then(|s| s.to_str()) == Some("wasm") {
        let path = path.with_extension("wat");
        log::debug!("writing file {}", path.display());
        std::fs::write(path, wasmprinter::print_bytes(&contents).unwrap()).unwrap();
    }
}
