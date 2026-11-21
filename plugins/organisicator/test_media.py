"""python3 -m unittest discover -s plugins/organisicator -p 'test_*.py'

The media browser's lists, from answers shaped like organisicator's API."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import media  # noqa: E402

INSTANCES = [
    {"id": 7, "parent": None, "value": "coffee", "code": "cf", "rel": "sequences/coffee"},
    {"id": 9, "parent": 7, "value": "cf_020", "code": "", "rel": "sequences/coffee/cf_020"},
    {"id": 8, "parent": 7, "value": "cf_010", "code": "", "rel": "sequences/coffee/cf_010"},
]
ANSWER = {
    "indexed": True,
    "assets": [
        {"id": 42, "kind": "sequence", "name": "bg.####.exr", "ext": "exr", "rel": "plates/bg/bg.####.exr", "frames": 80},
        {"id": 43, "kind": "file", "name": "notes.pdf", "ext": "pdf", "rel": "notes.pdf", "frames": None},
    ],
    "linked": [
        {"id": 50, "kind": "file", "rel": "sequences/coffee/assets/smoke.mov", "path": "/p/sequences/coffee/assets/smoke.mov",
         "via": "sequences/coffee", "exists": True},
        {"id": None, "kind": "sequence", "rel": "library/clean.####.exr", "path": "/p/library/clean.####.exr",
         "via": "", "exists": True},
    ],
}


class Media(unittest.TestCase):
    def test_shots_in_tree_order_with_the_project_first(self):
        opts = media.shot_options(INSTANCES)
        self.assertEqual(opts, [("(project)", None), ("coffee", 7), ("coffee › cf_010", 8), ("coffee › cf_020", 9)])

    def test_own_then_linked_footage_only_what_a_load_opens(self):
        rows = media.footage_rows(ANSWER, INSTANCES)
        self.assertEqual([r["rel"] for r in rows], ["plates/bg/bg.####.exr", "sequences/coffee/assets/smoke.mov", "library/clean.####.exr"])
        self.assertIn("80f", rows[0]["label"])
        self.assertIn("linked via coffee", rows[1]["label"])
        self.assertIn("linked via the project", rows[2]["label"])
        self.assertEqual(len({r["label"] for r in rows}), 3, "labels are unique")
        self.assertIsNone(rows[2]["id"])
        self.assertEqual(rows[2]["path"], "/p/library/clean.####.exr")

    def test_filter_and_info(self):
        rows = media.footage_rows(ANSWER, INSTANCES, "SMOKE")
        self.assertEqual([r["rel"] for r in rows], ["sequences/coffee/assets/smoke.mov"])
        self.assertIn("by organisicator id", media.info(rows[0], ANSWER))
        self.assertIn("by path", media.info(media.footage_rows(ANSWER, INSTANCES, "clean")[0], ANSWER))
        self.assertIn("not indexed", media.info(None, {"indexed": False}))

    def test_locate_a_loads_asset(self):
        a = {"id": 42, "project": {"id": 5, "name": "spooky"}, "shot": [{"id": 7, "value": "coffee"}, {"id": 8, "value": "cf_010"}]}
        self.assertEqual(media.locate(a), (5, 8))
        self.assertEqual(media.locate({"project": {"id": 5}, "shot": []}), (5, None))



ALL = {
    "indexed": True,
    "assets": [
        {"id": 1, "kind": "sequence", "rel": "sequences/asset/renders/light/light_####.png", "frames": 24},
        {"id": 2, "kind": "file", "rel": "sequences/asset/renders/still.png"},
        {"id": 3, "kind": "file", "rel": "sequences/asset/assets/tree.exr"},
        {"id": 4, "kind": "file", "rel": "reference/moodboard.jpg"},
        {"id": 5, "kind": "file", "rel": "reference/notes.pdf"},
        {"id": 6, "kind": "file", "rel": "docs/brief.pdf"},
        {"id": 7, "kind": "file", "rel": "other/renders/x.png"},
    ],
}


class Tree(unittest.TestCase):
    def test_only_folders_with_loadable_footage(self):
        c = media.folder_counts(ALL)
        self.assertNotIn("docs", c, "a pdf isn't footage")
        self.assertEqual(c[""], (0, 5))
        self.assertEqual(c["sequences/asset/renders"], (1, 2))
        self.assertEqual(c["sequences/asset"], (0, 3))
        self.assertEqual(c["reference"], (1, 1))

    def test_rows_follow_what_is_expanded(self):
        c = media.folder_counts(ALL)
        rows = media.tree_rows(c, {""})
        self.assertEqual([d for _, d in rows], ["", "other", "reference", "sequences"])
        self.assertTrue(rows[0][0].startswith("▾ (project)"))
        exp = media.expand_to("sequences/asset/renders", {""})
        rows = media.tree_rows(c, exp)
        self.assertIn("sequences/asset/renders/light", [d for _, d in rows])
        # Two folders called "renders": still one label each.
        exp |= {"other"}
        labels = [l for l, _ in media.tree_rows(c, exp)]
        self.assertEqual(len(labels), len(set(labels)))

    def test_a_folder_and_recursive(self):
        own = media.under(ALL, "sequences/asset/renders", False)
        self.assertEqual([a["rel"] for a in own["assets"]], ["still.png"])
        rec = media.under(ALL, "sequences/asset", True)
        self.assertEqual([a["rel"] for a in rec["assets"]],
                         ["renders/light/light_####.png", "renders/still.png", "assets/tree.exr"])
        self.assertEqual(len(media.under(ALL, "", True)["assets"]), 7)
        rows = media.footage_rows(rec, [])
        self.assertEqual(rows[0]["id"], 1)

    def test_where_things_are(self):
        self.assertEqual(media.folder_of("/m/p/sequences/a/renders/l/l_####.png", "/m/p/"), "sequences/a/renders/l")
        self.assertEqual(media.folder_of("D:\\m\\p\\x.exr", "D:\\m\\p"), "")
        self.assertEqual(media.folder_of("/else/x.exr", "/m/p"), "")
        inst = [{"id": 7, "rel": "sequences/asset"}]
        self.assertEqual(media.instance_at("sequences/asset", inst), 7)
        self.assertIsNone(media.instance_at("sequences/asset/renders", inst))

if __name__ == "__main__":
    unittest.main()
