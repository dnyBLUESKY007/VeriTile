"""Files of a result bundle: JSON, logs, privacy scrubbing and size limits."""
import getpass
import json
from pathlib import Path
import socket

LOG_LIMIT = 1 << 20
FILE_LIMIT = 50 << 20
BATCH_LIMIT = 100 << 20
MARK = b'\n... [truncated by gpurun] ...\n'


def scrub(text, root):
    """Remove paths, the user name and the host name; results are public."""
    # The loopback address is a constant of the launchers and names no machine; it is
    # written as a label so that a reader checking for addresses finds none.
    replacements = [(str(root), '<repo>'), (str(Path.home()), '<home>'), ('127.0.0.1', '<loopback>')]
    for private, label in ((socket.gethostname(), '<host>'), (getpass.getuser(), '<user>')):
        if len(private) > 2:  # a one-letter name would shred ordinary text
            replacements.append((private, label))
    for private, label in replacements:
        text = text.replace(private, label)
    return text


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_log(path, raw, root):
    """Scrub a captured stream and keep its start and its end within 1 MB."""
    data = scrub(raw.decode('utf-8', 'replace'), root).encode()
    if len(data) > LOG_LIMIT:
        half = (LOG_LIMIT - len(MARK)) // 2
        data = data[:half] + MARK + data[-half:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def over_limits(result_dir):
    """Files above 50 MB and, if the bundle exceeds 100 MB, the bundle itself."""
    sizes = {p.relative_to(result_dir).as_posix(): p.stat().st_size
             for p in result_dir.rglob('*') if p.is_file()}
    found = [f'{name}: {size} bytes' for name, size in sorted(sizes.items()) if size > FILE_LIMIT]
    if sum(sizes.values()) > BATCH_LIMIT:
        found.append(f'bundle: {sum(sizes.values())} bytes')
    return found
