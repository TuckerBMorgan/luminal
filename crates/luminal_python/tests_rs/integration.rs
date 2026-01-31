#[test]
fn python_tests_pass() {
    let output = match std::process::Command::new("pytest")
        .args(["tests", "-v", "--tb=short"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
    {
        Ok(output) => output,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("pytest not found, skipping Python tests");
            eprintln!("Run 'uv pip install pytest' in a virtual environment to enable");
            return;
        }
        Err(e) => panic!("Failed to run pytest: {}", e),
    };

    if !output.status.success() {
        eprintln!("STDOUT: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Python tests failed");
    }
}
