"""python3 -m unittest discover -s plugins/organisicator -p 'test_*.py'

A throwaway managed root holding one project, its manifest as organisicator
writes it (the Blender add-on's fixture, with a comp folder added)."""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core  # noqa: E402
import shots  # noqa: E402

MANIFEST = {
    "format": 1, "app": "organisicator", "name": "spooky_beach", "code": "spb", "anatomy": "VFX Show",
    "folders": [
        {"path": "sequences/{sequence}", "type": "sequence"},
        {"path": "sequences/{sequence}/{shot}", "type": "shot"},
        {"path": "sequences/{sequence}/{shot}/renders", "type": "render", "exts": "exr mov"},
        {"path": "sequences/{sequence}/{shot}/scripts", "type": "software", "exts": "nk blend wbl"},
        {"path": "sequences/{sequence}/{shot}/plates", "type": "source", "exts": "exr"},
    ],
    "levels": [],
    "instances": [
        {"id": 1, "parent": None, "decl": "sequences/{sequence}", "value": "dev", "code": "dv", "rel": "sequences/dev"},
        {"id": 2, "parent": 1, "decl": "sequences/{sequence}/{shot}", "value": "spb_010", "code": "", "rel": "sequences/dev/spb_010"},
        {"id": 3, "parent": 1, "decl": "sequences/{sequence}/{shot}", "value": "spb_020", "code": "", "rel": "sequences/dev/spb_020"},
    ],
}


class Shots(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.proj_dir = os.path.join(self.root, "live", "spooky_beach")
        os.makedirs(os.path.join(self.proj_dir, ".organisicator"))
        with open(os.path.join(self.proj_dir, ".organisicator", "project.json"), "w") as fh:
            json.dump(MANIFEST, fh)
        self.shot_dir = os.path.join(self.proj_dir, "sequences", "dev", "spb_010")
        for sub in ("renders", "scripts", "plates/bg", "plates/fg"):
            os.makedirs(os.path.join(self.shot_dir, sub))
        (self.proj,) = core.find_projects(self.root)
        self.shot = self.proj.instance(2)

    def tearDown(self):
        shutil.rmtree(self.root)

    def touch(self, *parts):
        p = os.path.join(self.shot_dir, *parts)
        open(p, "w").close()
        return p

    def test_labels_walk_the_tree(self):
        labels = [lbl for lbl, _ in shots.shot_labels(self.proj)]
        self.assertEqual(labels, ["(project)", "dev", "dev / spb_010", "dev / spb_020"])

    def test_comps_go_in_the_software_folder_or_its_wobble_child(self):
        scripts = os.path.join(self.shot_dir, "scripts")
        self.assertEqual(shots.comp_dir(self.proj, self.shot), scripts)
        os.makedirs(os.path.join(scripts, "wobble"))
        self.assertEqual(shots.comp_dir(self.proj, self.shot), os.path.join(scripts, "wobble"))

    def test_new_comps_are_named_from_the_shot_and_never_overwrite(self):
        p = shots.new_comp_path(self.proj, self.shot, "comp")
        self.assertEqual(os.path.basename(p), "spb_010_comp_v001.wbl")
        open(p, "w").close()
        self.assertEqual(os.path.basename(shots.new_comp_path(self.proj, self.shot, "comp")), "spb_010_comp_v002.wbl")
        self.assertEqual(os.path.basename(shots.version_up(p)), "spb_010_comp_v002.wbl")

    def test_comp_files_are_newest_first(self):
        a = self.touch("scripts", "a_v001.wbl")
        b = self.touch("scripts", "a_v002.wbl")
        self.touch("scripts", "notes.txt")
        old = time.time() - 100
        os.utime(a, (old, old))
        self.assertEqual(shots.comp_files(os.path.dirname(a)), [b, a])

    def test_writes_land_in_the_shot_renders_one_folder_each(self):
        renders = os.path.join(self.shot_dir, "renders")
        one = shots.write_targets(self.proj, self.shot, "spb_010_comp_v003", ["Write"])
        self.assertEqual(one["Write"], os.path.join(renders, "spb_010_comp_v003", "spb_010_comp_v003.####.exr"))
        two = shots.write_targets(self.proj, self.shot, "c", ["main", "matte out"])
        self.assertEqual(two["matte out"], os.path.join(renders, "c_matte_out", "c_matte_out.####.exr"))
        self.assertNotEqual(os.path.dirname(two["main"]), os.path.dirname(two["matte out"]))

    def test_plates_collapse_to_one_read_path_per_sequence(self):
        for f in range(1001, 1004):
            self.touch("plates", "bg", f"bg.{f}.exr")
            self.touch("plates", "fg", f"fg_v2_{f:05d}.exr")
        self.touch("plates", "ref.mov")
        self.touch("plates", "bg", ".DS_Store")
        plates = os.path.join(self.shot_dir, "plates")
        self.assertEqual(shots.plates_dir(self.proj, self.shot), plates)
        self.assertEqual(shots.sequences(plates), sorted([
            os.path.join(plates, "bg", "bg.####.exr"),
            os.path.join(plates, "fg", "fg_v2_#####.exr"),
            os.path.join(plates, "ref.mov"),
        ]))

    def test_plates_become_organisicator_asset_ids(self):
        bg = os.path.join(self.shot_dir, "plates", "bg", "bg.####.exr")
        fg = os.path.join(self.shot_dir, "plates", "fg", "fg.####.exr")
        projects = [{"id": 5, "root": self.proj_dir}]
        listing = {"indexed": True, "assets": [{"id": 42, "rel": "plates/bg/bg.####.exr"}]}
        self.assertEqual(shots.asset_ids(self.proj, self.shot, [bg, fg], projects, listing), {bg: 42})
        self.assertEqual(shots.asset_ids(self.proj, self.shot, [bg], [{"id": 1, "root": "/elsewhere"}], listing), {})

    def test_the_open_comp_finds_its_shot(self):
        p = self.touch("scripts", "spb_010_comp_v001.wbl")
        proj, inst = core.locate([self.proj], p)
        self.assertEqual((proj.name, inst.value), ("spooky_beach", "spb_010"))

    def test_core_is_organisicators_own_copy(self):
        """core.py is copied from organisicator's Blender add-on; when that
        checkout is next to wobble, the copy must match it."""
        here = os.path.dirname(os.path.abspath(__file__))
        theirs = os.path.join(here, "..", "..", "..", "organisicator", "blender_addon", "organisicator", "core.py")
        if not os.path.isfile(theirs):
            self.skipTest("no organisicator checkout beside wobble")
        with open(theirs) as a, open(os.path.join(here, "core.py")) as b:
            self.assertEqual(a.read(), b.read(), "re-copy core.py from organisicator")


if __name__ == "__main__":
    unittest.main()
