"""Generate the diagnosis page and embed it into the generated dashboard."""
import hashlib
import json
from .run import HERE, report
from tests.progress_dashboard.build import main, TESTS

report()
main()
page=TESTS/'index.html'
detail=(HERE/'assets/index.html').read_text()
body=detail.split('</style>',1)[1]
block='<section id="p2-fit-diagnostic"><details><summary>P2最新：时间相关分解与固定小样本拟合（点击展开）</summary>'+body+'</details></section>'
text=page.read_text()
assert '</main>' in text
page.write_text(text.replace('</main>',block+'</main>',1))
snapshot=TESTS/'progress_dashboard/snapshot.json'
record=json.loads(snapshot.read_text())
record['sources'][str((HERE/'assets/index.html').relative_to(TESTS.parent))]=hashlib.sha256((HERE/'assets/index.html').read_bytes()).hexdigest()
snapshot.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
