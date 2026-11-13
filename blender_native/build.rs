fn main() {
    // A Python extension module leaves the interpreter's symbols
    // unresolved until Blender loads it: the linker must allow that.
    pyo3_build_config::add_extension_module_link_args();
}
