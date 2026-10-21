"""Render the packaged Markdown and check public GitHub resource URLs anonymously."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import re
import urllib.request
import zipfile

from PIL import Image
from readme_renderer.markdown import render


class ResourceLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("src", "href") and value:
                self.links.append(value)


def check(dist: Path, preview: Path, *, network: bool = False) -> dict:
    wheel = next(dist.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        data = archive.read(name)
        text = data.replace(b"\r\n", b"\n").partition(b"\n\n")[2].decode("utf-8")
    parser = ResourceLinks()
    parser.feed(text)
    links = parser.links + re.findall(r"(?<!!)\[[^\]]+\]\(([^)\s]+)\)", text)
    assert all(link.startswith(("https://", "#")) for link in links), "Relative or insecure README resource link"
    html = render(text)
    assert html and "Quick Test" in html and "Full Evaluation" in html
    preview.write_text("<!doctype html><meta charset='utf-8'><title>SciFigBench PyPI preview</title>\n" + html,
                       encoding="utf-8")
    resources = sorted({link for link in links if link.startswith((
        "https://github.com/LeNcyer/SciFigBench/blob/main/",
        "https://raw.githubusercontent.com/LeNcyer/SciFigBench/main/",
    ))})
    required = ("assets/benchmark_overview.png", "assets/SciFigBench%20Poster.pdf", "LICENSE", "AGENTS.md")
    assert all(any(url.endswith(suffix) for url in resources) for suffix in required), "Missing required README resource"
    assert all(url in html for url in resources), "Rendered description lost a required resource link"
    results = []
    if network:
        # urllib does not use the user's browser session or Git credentials.
        opener = urllib.request.build_opener()
        for url in resources:
            request = urllib.request.Request(url, headers={"User-Agent": "SciFigBench-release-check"})
            with opener.open(request, timeout=60) as response:
                data = response.read()
                mime = response.headers.get_content_type()
                assert response.status == 200, f"Resource unavailable: {url}"
                if url.endswith(".png"):
                    assert mime == "image/png"
                    with Image.open(BytesIO(data)) as image:
                        image.verify()
                results.append({"url": url, "status": 200, "content_type": mime, "bytes": len(data)})
    return {"rendered": True, "preview": str(preview), "relative_resource_links": 0,
            "anonymous_https_checked": network, "resource_count": len(resources), "urls": resources, "resources": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--network", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.dist, args.preview, network=args.network), indent=2))
