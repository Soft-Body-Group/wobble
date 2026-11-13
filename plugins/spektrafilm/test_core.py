"""python3 -m unittest test_core   (from this folder). The bake tests need the
baker built (lutgen/) and spektrafilm-rs's data; they skip without them."""

import os
import unittest

import core


class Core(unittest.TestCase):
    def test_spaces(self):
        self.assertEqual(core.space_of("Colorspace { primaries: Ap1, curve: Linear }"), "ACEScg")
        self.assertEqual(core.space_of("Colorspace { primaries: Rec709, curve: Linear }"), "ITU-R BT.709")
        with self.assertRaises(ValueError):
            core.space_of("Colorspace { primaries: RedWideGamut, curve: Linear }")

    def test_params(self):
        r = core.default_recipe()
        core.set_param(r, "enlarger.print_exposure", 1.5)
        core.set_param(r, "enlarger.m_filter_shift", -3.0)
        self.assertEqual(core.get_param(r, "enlarger.print_exposure", 1.0), 1.5)
        self.assertEqual(core.get_param(r, "enlarger.y_filter_shift", 0.0), 0.0)
        self.assertEqual(r["params"], {"enlarger": {"print_exposure": 1.5, "m_filter_shift": -3.0}})

    def test_recipe_line(self):
        src = '//! name: x\n// spektrafilm: {"baker":1,"film":"kodak_gold_200","size":17}\nfn pass1() {}'
        self.assertEqual(core.recipe_of(src), {"film": "kodak_gold_200", "size": 17})
        self.assertIsNone(core.recipe_of("//! name: Shader\n"))

    def test_label(self):
        films = [{"id": "kodak_portra_400", "label": "Kodak Portra 400", "print": "kodak_portra_endura"}]
        papers = [{"id": "kodak_portra_endura", "label": "Kodak Portra Endura"}]
        r = core.default_recipe()
        self.assertEqual(core.label_of(r, films, papers), "Portra 400 / Portra Endura")
        r["scan_film"] = True
        self.assertEqual(core.label_of(r, films, papers), "Portra 400")


@unittest.skipUnless(core.baker_ready() and os.path.isdir(core.data_dir()), "baker or spektrafilm data missing")
class Bake(unittest.TestCase):
    def test_profiles(self):
        p = core.profiles()
        ids = [f["id"] for f in p["films"]]
        self.assertIn("kodak_portra_400", ids)
        self.assertTrue(p["papers"])

    def test_bake_round_trips_the_recipe(self):
        r = core.default_recipe()
        r["size"] = 17
        core.set_param(r, "enlarger.print_exposure", 1.2)
        src = core.bake(r)
        self.assertTrue(src.startswith("//! name: "))
        self.assertIn("//! param: float exposure", src)
        back = core.recipe_of(src)
        self.assertEqual(back["film"], "kodak_portra_400")
        self.assertEqual(back["params"]["enlarger"]["print_exposure"], 1.2)
        self.assertIn("array<u32, %d>" % (2 * 17 ** 3), src)

    def test_unknown_stock_says_so(self):
        r = core.default_recipe("no_such_film")
        with self.assertRaises(RuntimeError) as e:
            core.bake(r)
        self.assertIn("no_such_film", str(e.exception))


if __name__ == "__main__":
    unittest.main()
