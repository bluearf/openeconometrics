"""Execute the deployment gate against the current native result protocol."""

import ast
from pathlib import Path
import textwrap

import openecon as oe
from openecon.output_latex import PUBLICATION_STYLE, enrich_record


def test_control_image_accepts_current_native_publication_result():
    config = (Path(__file__).parents[1] / "deploy/cloudbuild-control.yaml").read_text()
    # Run the actual archive-compatibility part of the image gate. Linux image
    # ownership/rootfs checks run inside Cloud Build, not this source-level test.
    block = config.split("        import json\n", 1)[1].split("        print('Production image:", 1)[0]
    gate = ast.parse(textwrap.dedent("        import json\n" + block))
    model = oe.ols(data=oe.example(), y="wage", x=["education", "experience"], covariance="HC3")
    namespace = {"model": model, "enrich_record": enrich_record, "PUBLICATION_STYLE": PUBLICATION_STYLE}
    exec(compile(gate, "cloudbuild-control.yaml:result-contract", "exec"), namespace)
