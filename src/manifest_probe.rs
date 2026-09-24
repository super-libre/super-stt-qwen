// SPDX-License-Identifier: GPL-3.0-only
//! Reading fields back out of `backend.toml`, for the tests that hold this
//! crate's tables to the manifest's.
//!
//! Parsed rather than deserialized because the manifest is the schema's shape
//! and not this crate's: pulling in a TOML dependency to read a handful of
//! fields would make the tests depend on a type they do not own.

/// The lines of every `[[models]]` block, in order. A block ends at the next
/// table header or at the end of the file.
pub(crate) fn model_blocks(manifest: &str) -> Vec<Vec<&str>> {
    manifest
        .split("[[models]]")
        .skip(1)
        .map(|block| {
            block
                .lines()
                .take_while(|l| !l.trim_start().starts_with('['))
                .collect()
        })
        .collect()
}

/// One `key = "value"` from a block, unquoted.
pub(crate) fn string_field<'a>(block: &[&'a str], key: &str) -> Option<&'a str> {
    block.iter().find_map(|line| {
        let (found, value) = line.trim().split_once('=')?;
        (found.trim() == key).then(|| value.trim().trim_matches('"'))
    })
}

/// A `key = [ "a", "b", … ]` list of strings from a block, written on one line
/// or one entry to a line.
pub(crate) fn string_list<'a>(block: &[&'a str], key: &str) -> Option<Vec<&'a str>> {
    let opens = block.iter().position(|line| {
        line.trim()
            .split_once('=')
            .is_some_and(|(found, _)| found.trim() == key)
    })?;
    let mut entries = Vec::new();
    for (i, line) in block[opens..].iter().enumerate() {
        let line = if i == 0 {
            line.split_once('=').map_or(*line, |(_, rest)| rest)
        } else {
            line
        };
        let line = line.split_once('[').map_or(line, |(_, rest)| rest);
        let (line, closes) = line
            .split_once(']')
            .map_or((line, false), |(entries, _)| (entries, true));
        entries.extend(
            line.split(',')
                .map(|c| c.trim().trim_matches('"'))
                .filter(|c| !c.is_empty()),
        );
        if closes {
            return Some(entries);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_read_on_one_line_or_many() {
        let manifest = "[[models]]\nname = \"a\"\nxs = [\"x\", \"y\"]\nys = [\n  \"p\",\n  \"q\",\n]\n[[models]]\nname = \"b\"\n";
        let blocks = model_blocks(manifest);
        assert_eq!(blocks.len(), 2);
        assert_eq!(string_field(&blocks[0], "name"), Some("a"));
        assert_eq!(string_list(&blocks[0], "xs"), Some(vec!["x", "y"]));
        assert_eq!(string_list(&blocks[0], "ys"), Some(vec!["p", "q"]));
        assert_eq!(string_field(&blocks[1], "name"), Some("b"));
    }
}
