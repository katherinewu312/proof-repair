#!/usr/bin/env python3
"""Extract all formalized OpenAI math results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mask_source(text: str) -> str:
    """Hide nested comments and strings while preserving character offsets/lines."""
    out = list(text)
    i = 0
    depth = 0
    string = False
    while i < len(text):
        if depth:
            if text.startswith('/-', i):
                out[i:i+2] = '  '; depth += 1; i += 2; continue
            if text.startswith('-/', i):
                out[i:i+2] = '  '; depth -= 1; i += 2; continue
        elif string:
            if text[i] == '\\':
                out[i] = ' '; i += 1
                if i < len(text):
                    out[i] = '\n' if text[i] == '\n' else ' '
                i += 1; continue
            if text[i] == '"':
                string = False
        elif text.startswith('--', i):
            end = text.find('\n', i)
            if end < 0: end = len(text)
            out[i:end] = ' ' * (end-i); i = end; continue
        elif text.startswith('/-', i):
            out[i:i+2] = '  '; depth = 1; i += 2; continue
        elif text[i] == '"':
            string = True
        else:
            i += 1; continue
        if text[i] != '\n': out[i] = ' '
        i += 1
    if depth or string:
        raise ValueError('Unterminated comment or string')
    return ''.join(out)


IDENT = r'(?:«[^»]+»|[\w\u0080-\uffff][\w\u0080-\uffff\'.]*)(?:\.(?:«[^»]+»|[\w\u0080-\uffff][\w\u0080-\uffff\']*))*'
COMMAND = re.compile(
    r'^(?P<indent>[ \t]*)(?:@\[[^\n]*?\]\s*)?'
    r'(?:(?:private|protected|noncomputable|unsafe|partial)\s+)*'
    r'(?P<kind>theorem|lemma|def|abbrev|opaque|axiom|instance|example|'
    r'namespace|section|end|open|variable|variables|include|omit|attribute|'
    r'set_option|notation|local|scoped|syntax|macro|elab|structure|class|inductive|'
    r'mutual|universe|universes|export|initialize|import)\b'
    r'(?:[ \t]+(?P<name>' + IDENT + r'))?', re.MULTILINE)


def locate(text: str, target: str) -> dict:
    masked = mask_source(text)
    commands = list(COMMAND.finditer(masked))
    stack: list[tuple[str, str]] = []
    matches = []
    for index, command in enumerate(commands):
        kind, name = command['kind'], command['name']
        if name:
            name = name.rstrip('.')
        if kind in ('namespace', 'section'):
            if kind == 'namespace' and name:
                stack.extend((kind, part) for part in name.split('.'))
            else:
                stack.append((kind, name or ''))
            continue
        if kind == 'end':
            for _ in range(len(name.split('.')) if name else 1):
                if stack:
                    stack.pop()
            continue
        if kind not in ('theorem', 'lemma', 'def', 'abbrev', 'opaque') or not name:
            continue
        namespace = '.'.join(n for k, n in stack if k == 'namespace' and n)
        full_name = name[len('_root_.'):] if name.startswith('_root_.') else '.'.join(filter(None, [namespace, name]))
        if full_name != target:
            continue
        start = command.start()
        end = len(text)
        for following in commands[index+1:]:
            if len(following['indent']) <= len(command['indent']):
                end = following.start(); break
        while end > start and text[end-1].isspace(): end -= 1
        # Find the declaration's assignment outside parameter/type delimiters.
        depth = 0
        assignment = None
        for pos in range(command.end(), end):
            c = masked[pos]
            if c in '([{⦃': depth += 1
            elif c in ')]}⦄': depth -= 1
            if depth == 0 and masked.startswith(':=', pos):
                assignment = pos; break
        if assignment is None:
            raise ValueError('Could not locate a declaration-level := proof assignment')
        proof_start = assignment + 2
        while proof_start < end and text[proof_start].isspace(): proof_start += 1
        def span(a, b):
            return {'start_byte': len(text[:a].encode()), 'end_byte': len(text[:b].encode()),
                    'start_line': text.count('\n', 0, a)+1, 'end_line': text.count('\n', 0, b)+1}
        matches.append({'declaration_kind': kind, 'declaration_text': text[start:end],
                        'statement_text': text[start:assignment].rstrip(),
                        'proof_text': text[proof_start:end],
                        'declaration_span': span(start, end), 'proof_span': span(proof_start, end),
                        'contains_sorry_or_admit': bool(re.search(r'\b(sorry|admit)\b', masked[proof_start:end]))})
    if len(matches) != 1:
        raise ValueError(f'Expected one source match for {target}; found {len(matches)}')
    return matches[0]


def git(project: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(project), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=Path('external/openai-math/lean'))
    parser.add_argument('--output', type=Path, default=Path('data/raw/openai_math/originals.jsonl'))
    parser.add_argument('--allow-dirty', action='store_true', help='Record a dirty checkout instead of rejecting it')
    args = parser.parse_args()
    project = args.project_dir.resolve()
    catalogue_path = project / 'formalization.yaml'
    catalogue_bytes = catalogue_path.read_bytes()
    targets = yaml.safe_load(catalogue_bytes)['status']['main_results']
    commit = git(project, 'rev-parse', 'HEAD')
    dirty = bool(git(project, 'status', '--porcelain', '--untracked-files=normal'))
    if dirty and not args.allow_dirty:
        raise ValueError('Source checkout has changes; commit them or pass --allow-dirty')
    fingerprint = hashlib.sha256()
    for name in ('lean-toolchain', 'lakefile.lean', 'lakefile.toml', 'lake-manifest.json'):
        path = project / name
        if path.is_file(): fingerprint.update(name.encode() + b'\0' + path.read_bytes())
    records, rejected = [], []
    for entry in targets:
        try:
            source_path = (project / entry['file']).resolve()
            source_path.relative_to(project)
            raw = source_path.read_bytes()
            text = raw.decode('utf-8')
            extracted = locate(text, entry['declaration'])
            config_path = (project / entry['comparator_config']).resolve()
            config_path.relative_to(project)
            comparator = json.loads(config_path.read_text())
            src_hash = digest((commit + '\0' + entry['file'] + '\0' + entry['declaration'] + '\0').encode() + raw)
            records.append({
                'selected_id': 'openai-original-' + src_hash,
                'source': 'openai_math', 'source_commit': commit, 'source_dirty': dirty,
                'source_file': entry['file'], 'theorem': entry['declaration'], 'src_hash': src_hash,
                'file_sha256': digest(raw), 'correct_proof': text,
                'lean_toolchain': (project / 'lean-toolchain').read_text().strip(),
                'verifier_fingerprint': fingerprint.hexdigest(),
                'comparator_config': entry['comparator_config'], 'comparator_config_contents': comparator,
                'extraction_method': 'lexical_namespace_match', 'verification_status': 'unverified',
                **extracted,
            })
        except (ValueError, OSError, KeyError) as error:
            rejected.append({**entry, 'reason': str(error)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    print(f'Extracted {len(records)}/{len(targets)} catalogue entries to {args.output}')
    print(f'Skipped: {len(rejected)}. All records are unverified.')
    for entry in rejected:
        print(f"Skipped {entry['declaration']}: {entry['reason']}", file=sys.stderr)
    return 1 if rejected else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(1)
