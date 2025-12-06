find . -print | while read -r path; do
    depth=$(( $(grep -o "/" <<< "$path" | wc -l) - 1 ))
    indent=$(printf '%*s' $((depth * 4)) "")
    if [ -d "$path" ]; then
        printf "%s%s/\n" "$indent" "${path##*/}"
    elif [[ "$path" == *.py ]]; then
        printf "%s%s\n" "$indent" "${path##*/}"
        grep -E "^[[:space:]]*(class|def) " "$path" | sed "s/^/$indent    /"
    fi
done

