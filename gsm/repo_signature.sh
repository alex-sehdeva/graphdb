find . -name "*.py" -exec bash -c '
    f="$1"
    printf "%s\n" "$f"
    grep -E "^[[:space:]]*(class|def) " "$f" | sed "s/^/    /"
' bash {} \;
