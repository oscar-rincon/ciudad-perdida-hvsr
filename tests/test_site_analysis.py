import ast
import json
from pathlib import Path
import unittest

from utils import site_analysis as sites

ROOT = Path(__file__).resolve().parents[1]


class SiteAnalysisTests(unittest.TestCase):
    def test_site_registry_has_15_unique_numbered_sessions(self):
        self.assertEqual(len(sites.SITES), 15)
        self.assertEqual(len({site.key for site in sites.SITES}), 15)
        self.assertEqual(len({site.session_id for site in sites.SITES}), 15)
        self.assertEqual(tuple(site.key[:2] for site in sites.SITES),
                         tuple(f"{i:02}" for i in range(1, 16)))
        self.assertEqual(set(sites.SITE_BY_KEY), {site.key for site in sites.SITES})


    def test_default_profile_matches_shared_terrace_processing(self):
        params = sites.BASE_PARAMS
        self.assertEqual(params.window_length_s, 120.0)
        self.assertEqual(params.window_selection, "continuous")
        self.assertEqual(params.overlap_pct, 0.0)
        self.assertEqual(params.sta_lta_method, "vector")
        self.assertEqual(params.smoothing_bandwidth, 40.0)
        self.assertEqual(params.horizontal_combination, "geometric_mean")
        self.assertEqual(params.peak_range, (0.1, 50.0))
        self.assertTrue(params.frequency_domain_rejection)

    def test_processing_folder_has_16_numbered_heading_only_notebooks(self):
        folder = ROOT / "main/02_hvsr_processing"
        notebooks = sorted(folder.glob("*.ipynb"))
        self.assertEqual(len(notebooks), 16)
        self.assertTrue(notebooks[-1].name.startswith("16_"))
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            markdown = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
            self.assertTrue(markdown)
            self.assertTrue(all(
                line.startswith("#")
                for cell in markdown
                for line in cell["source"]
                if line.strip()
            ))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]), filename=str(path))
        for workflow in ("01_data_organization_hvsr", "02_hvsr_processing"):
            for output in ("data", "figures"):
                self.assertTrue((ROOT / "main" / workflow / output).is_dir())
