find . -name "*.py" -print -exec awk '
  /^[[:space:]]*(class|def) / { printf "    %s\n", $0 }
' {} \;

