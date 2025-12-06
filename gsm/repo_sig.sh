#!/usr/bin/env bash
set -euo pipefail

# Default: include triple-quoted docstrings (module, class, def)
INCLUDE_DOCS="${INCLUDE_DOCS:-true}"

find . -name "*.py" -print0 | while IFS= read -r -d '' f; do
    # Print the relative path
    printf '%s\n' "$f"

    awk -v include_docs="$INCLUDE_DOCS" '
        BEGIN {
            in_doc = 0              # inside a docstring we care about
            need_doc = 0            # just finished a header (def/class) looking for its docstring
            doc_context = ""        # "def", "class", or "module"
            in_def_header = 0       # currently printing a multi-line def header
            in_class_header = 0     # currently printing a multi-line class header
            module_doc_done = 0     # have we already printed the module-level docstring?
            have_seen_code = 0      # have we seen any real code yet (not comments/blank/doc)?
        }

        # --- def start line ---
        /^[[:space:]]*def[[:space:]]/ {
            print "    " $0
            in_def_header = 1
            have_seen_code = 1
            # header ends on this same line?
            if ($0 ~ /:[[:space:]]*$/) {
                in_def_header = 0
                if (include_docs == "true") {
                    need_doc = 1
                    doc_context = "def"
                }
            }
            next
        }

        # --- continuation of a multi-line def header ---
        in_def_header {
            print "    " $0
            have_seen_code = 1
            if ($0 ~ /:[[:space:]]*$/) {
                in_def_header = 0
                if (include_docs == "true") {
                    need_doc = 1
                    doc_context = "def"
                }
            }
            next
        }

        # --- class start line ---
        /^[[:space:]]*class[[:space:]]/ {
            print "    " $0
            in_class_header = 1
            have_seen_code = 1
            if ($0 ~ /:[[:space:]]*$/) {
                in_class_header = 0
                if (include_docs == "true") {
                    need_doc = 1
                    doc_context = "class"
                }
            }
            next
        }

        # --- continuation of a multi-line class header ---
        in_class_header {
            print "    " $0
            have_seen_code = 1
            if ($0 ~ /:[[:space:]]*$/) {
                in_class_header = 0
                if (include_docs == "true") {
                    need_doc = 1
                    doc_context = "class"
                }
            }
            next
        }

        {
            # --- module-level docstring detection (top of file) ---
            if (include_docs == "true" && module_doc_done == 0 && have_seen_code == 0) {
                # skip blank lines and comments before module doc
                if ($0 ~ /^[[:space:]]*$/ || $0 ~ /^[[:space:]]*#/) {
                    next
                }

                # first triple-quoted block before any code -> module docstring
                if ($0 ~ /^[[:space:]]*"""/) {
                    print "    " $0
                    in_doc = 1
                    doc_context = "module"
                    module_doc_done = 1

                    # single-line """doc""" case
                    n = gsub(/"""/, "&")
                    if (n >= 2) {
                        in_doc = 0
                        doc_context = ""
                    }
                    next
                }

                # first non-blank, non-comment, non-docstring line is real code
                have_seen_code = 1
                # fall through to normal handling for this line
            }

            if (include_docs == "true") {

                # --- docstring immediately after def/class header ---
                if (need_doc) {
                    # allow blank lines between header and docstring
                    if ($0 ~ /^[[:space:]]*$/) {
                        next
                    }

                    # first """ line after header
                    if ($0 ~ /^[[:space:]]*"""/) {
                        print "    " $0
                        in_doc = 1
                        # doc_context is already "def" or "class"
                        need_doc = 0

                        # handle """one-liner"""
                        n = gsub(/"""/, "&")
                        if (n >= 2) {
                            in_doc = 0
                            doc_context = ""
                        }
                        next
                    }

                    # some other non-blank, non-docstring line -> no docstring for this header
                    need_doc = 0
                    # fall through to possible other processing
                }

                # --- inside any tracked docstring (module/def/class) ---
                if (in_doc) {
                    print "    " $0
                    if (index($0, "\"\"\"") > 0) {
                        # closing line of the block
                        in_doc = 0
                        doc_context = ""
                    }
                }
            }
        }
    ' "$f"
done

