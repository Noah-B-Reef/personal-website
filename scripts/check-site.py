"""Check the Hugo migration's public routes, local assets, and math output."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
import xml.etree.ElementTree as ET
import re

ROOT = Path(__file__).resolve().parents[1] / "public"
BASE = re.search(r'^base-path = "([^"\n]+)"$', (ROOT.parent / "tola.toml").read_text(), re.M).group(1)
POSTS = {"hello-world": 0, "decision-tree-algorithms": 64, "k-nearest-neighbors": 75}


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links = []
        self.math = 0
        self.in_toc = False
        self.code = 0
        self.math_issues = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])
        if tag == "aside" and "toc" in attrs.get("class", "").split():
            self.in_toc = True
        self.math += attrs.get("role") == "math" and not self.in_toc
        self.code += tag == "pre"
        if "data-latex-issues" in attrs:
            self.math_issues.append(attrs["data-latex-issues"])

    def handle_endtag(self, tag):
        if tag == "aside":
            self.in_toc = False


required = ["index.html", "about/index.html", "blog/index.html", "tags/index.html",
            "search/index.html", "404.html", "resume.pdf", "images/perceptron.png", "images/photo.JPG",
            "feed.xml", "index.xml", "blog/index.xml", "sitemap.xml", "pagefind/pagefind.js"]
required += [f"tags/{tag}/index.html" for tag in (
    "machine-learning", "decision-trees", "statistics", "deep-learning",
    "neural-networks", "classification", "regression")]
for name in required:
    assert (ROOT / name).is_file(), f"Missing output: {name}"

pages = list(ROOT.rglob("*.html"))
for path in pages:
    page = Page(path.read_text())
    assert not page.math_issues, f"Math cannot copy as LaTeX: {path}: {page.math_issues}"
    current = BASE + path.relative_to(ROOT).as_posix()
    for link in page.links:
        if link.startswith("#"):
            continue
        url = urlsplit(urljoin(current, link))
        if url.scheme or url.netloc:
            continue
        assert url.path.startswith(BASE), f"Link escapes deployment path: {path}: {link}"
        target = ROOT / unquote(url.path.removeprefix(BASE))
        if url.path.endswith("/"):
            target /= "index.html"
        assert target.is_file(), f"Broken local link: {path}: {link}"

for slug, count in POSTS.items():
    page = Page((ROOT / "blog" / slug / "index.html").read_text())
    assert page.math == count, f"Equation count changed in {slug}: {page.math} != {count}"
    assert page.code >= 1, f"Missing code example in {slug}"

for name in ("feed.xml", "index.xml", "blog/index.xml", "sitemap.xml"):
    ET.parse(ROOT / name)
print(f"Verified {len(pages)} pages, all local links, legacy feeds, and 139 migrated equations.")
