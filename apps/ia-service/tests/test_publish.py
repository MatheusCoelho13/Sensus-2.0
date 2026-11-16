import json
import subprocess
import sys
import tempfile
from pathlib import Path

PUBLISH = Path(__file__).parent.parent / 'scripts' / 'publish.py'


def sh(*cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def setup():
    """'GitHub' falso (bare) + clone com apps/ia-service, *.pt ignorado como no repo real."""
    base = Path(tempfile.mkdtemp())
    sh('git', 'init', '-q', '--bare', '-b', 'main', str(base / 'remote.git'))
    repo = base / 'repo'
    sh('git', 'clone', '-q', str(base / 'remote.git'), str(repo))
    for k, v in (('user.name', 't'), ('user.email', 't@t')):
        sh('git', 'config', k, v, cwd=repo)
    svc = repo / 'apps' / 'ia-service'
    (svc / 'models').mkdir(parents=True)
    (repo / '.gitignore').write_text('*.pt\n')
    (repo / 'README.md').write_text('x')
    sh('git', 'add', '-A', cwd=repo)
    sh('git', 'commit', '-qm', 'init', cwd=repo)
    sh('git', 'push', '-q', 'origin', 'HEAD:main', cwd=repo)
    sh('git', 'branch', '-u', 'origin/main', cwd=repo)
    return base, repo, svc


def add_run(svc, map50=0.61, ts='20260101-120000'):
    (svc / 'models' / 'best.pt').write_bytes(b'pesos' * 10)
    d = svc / 'runs' / 'report' / ts
    d.mkdir(parents=True, exist_ok=True)
    (d / 'relatorio.html').write_text('<html>ok</html>')
    (d / 'relatorio.json').write_text(json.dumps({'summary': {'map50': map50}}))


def publish(svc, *extra):
    return sh(sys.executable, str(PUBLISH), '--root', str(svc), *extra)


def remote_files(base):
    return sh('git', '--git-dir', str(base / 'remote.git'), 'ls-tree', '-r', '--name-only', 'main').stdout.split()


def test_publishes_model_and_report_but_not_unrelated_changes():
    base, repo, svc = setup()
    add_run(svc)
    (repo / 'segredo.txt').write_text('nao subir')      # arquivo novo, não relacionado
    sh('git', 'add', 'segredo.txt', cwd=repo)           # ...até já no staging
    r = publish(svc)
    assert r.returncode == 0 and 'publicado' in r.stdout, r.stdout + r.stderr
    files = remote_files(base)
    assert 'apps/ia-service/models/best.pt' in files                      # *.pt ignorado, mas add -f
    assert 'apps/ia-service/reports/20260101-120000/relatorio.html' in files
    assert 'segredo.txt' not in files
    log = sh('git', '--git-dir', str(base / 'remote.git'), 'log', '-1', '--format=%s', 'main').stdout
    assert 'mAP50=0.610' in log


def test_second_run_has_nothing_new():
    base, repo, svc = setup()
    add_run(svc)
    publish(svc)
    r = publish(svc)
    assert 'nada novo' in r.stdout


def test_dry_run_changes_nothing():
    base, repo, svc = setup()
    add_run(svc)
    r = publish(svc, '--dry-run')
    assert 'dry-run' in r.stdout and 'apps/ia-service/models/best.pt' not in remote_files(base)


def test_refuses_missing_and_huge_model():
    base, repo, svc = setup()
    assert publish(svc).returncode != 0                      # sem best.pt
    add_run(svc)
    with open(svc / 'models' / 'best.pt', 'wb') as f:
        f.truncate(95 * 1024 * 1024)                         # arquivo esparso de 95MB
    r = publish(svc)
    assert r.returncode != 0 and 'gh release' in (r.stdout + r.stderr)
