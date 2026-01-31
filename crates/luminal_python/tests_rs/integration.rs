#[test]
fn python_tests_pass() {
    let output = std::process::Command::new("pytest")
        .args(["tests", "-v", "--tb=short"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to run pytest");

    if !output.status.success() {
        eprintln!("STDOUT: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Python tests failed");
    }
}
