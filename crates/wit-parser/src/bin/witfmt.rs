// use wit_parser::UnresolvedPackageGroup;

pub fn main() {
    for i in std::env::args().skip(1) {
        if i.starts_with("-") {
        } else {
            wit_parser::pretty_print(&i);
            // let contents = std::fs::read_to_string(&i).unwrap();
            // let tokens = wit_parser::ast::lex::Tokenizer::new(&contents, );
            // let x = UnresolvedPackageGroup::parse_file(&i);
            // match x {
            //     Ok(group) => {
            //         dbg!(group.main.name);
            //         dbg!(group.main.docs);
            //     }
            //     Err(e) => {
            //         dbg!(e);
            //     }
            // }
        }
    }
}
