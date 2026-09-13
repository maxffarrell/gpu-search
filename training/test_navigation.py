"""Parser contracts only: no provider evaluation, training, or holdout scoring."""
import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import training.prepare_navigation as navigation


class NavigationParserTests(unittest.TestCase):
    def test_english_fields_exclude_translations_and_other_sections(self):
        text = '''[Desktop Entry]
Name=Display
Name[fr]=Affichage
_Name[de]=Anzeige
Comment=Change display settings
Comment[fr]=Description traduite
Keywords=screen;monitor;
Keywords[fr]=écran;
[Desktop Action Custom]
Name=Unrelated action
Keywords=wrong;
'''
        fields, lines = navigation.english_desktop(text)
        self.assertEqual(fields['Name'], 'Display')
        self.assertEqual(fields['Comment'], 'Change display settings')
        self.assertEqual(fields['Keywords'], 'screen;monitor;')
        self.assertEqual(lines['Keywords'], 7)
        self.assertNotIn('Name[fr]', fields)
        fields, _ = navigation.english_desktop('[Desktop Entry]\nName[fr]=Écran\nKeywords[fr]=écran;\n')
        self.assertNotIn('Name', fields)
        self.assertNotIn('Keywords', fields)

    def test_gettext_markers_and_desktop_escapes(self):
        fields, _ = navigation.english_desktop('[Desktop Entry]\n_Name=Keyboard\n_Comment=Change keys\n_Keywords=keys;typing;\n')
        self.assertEqual(fields['Name'], 'Keyboard')
        self.assertEqual(navigation.split_keywords(r'keyboard;semi\;colon;space\sword;;', ';'), ['keyboard', 'semi;colon', 'space word'])
        self.assertEqual(navigation.split_keywords('Time, Date,Clock,,', ','), ['Time', 'Date', 'Clock'])

    def fixture_extract(self):
        temporary = tempfile.TemporaryDirectory(prefix='gpu-search-navigation-tests-')
        self.addCleanup(temporary.cleanup)
        root = pathlib.Path(temporary.name)
        for source in navigation.SOURCES:
            (root/'raw'/source['product']).mkdir(parents=True)
        def put(product, path, text):
            target = root/'raw'/product/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        put('gnome-control-center', 'COPYING', 'Test upstream GPL license evidence; preserved verbatim.\n')
        put('gnome-control-center', 'panels/display/display.desktop.in', '[Desktop Entry]\nName=Display\nName[fr]=Écran\nComment=Set display options\nKeywords=Screen;monitor;Screen;\nNoDisplay=true\nType=Application\n')
        put('gnome-control-center', 'panels/color/color.desktop.in', '[Desktop Entry]\nName=Color\nKeywords=screen;color;\n')
        put('plasma-desktop', 'LICENSES/LGPL-2.1-or-later.txt', 'Test preserved LGPL notice\n')
        put('plasma-desktop', 'kcms/clock/clock.json', json.dumps({'KPlugin': {'Name': 'Clock', 'Name[fr]': 'Horloge', 'Description': 'Set the time', 'Description[fr]': 'Traduction'}, 'X-KDE-Keywords': 'Time,Date', 'X-KDE-Keywords[fr]': 'heure,date'}))
        put('plasma-desktop', 'kcms/clock/clock.json.license', 'SPDX-License-Identifier: LGPL-2.1-or-later\n')
        put('xfce4-settings', 'COPYING', 'Test GPL copyright evidence\n')
        put('xfce4-settings', 'dialogs/keyboard-settings/keyboard.desktop.in', '# SPDX-License-Identifier: GPL-3.0-or-later\n[Desktop Entry]\nName=Keyboard\nKeywords=keys;\nHidden=false\n')
        put('xfce4-settings', 'dialogs/mime-settings/xfce4-web-browser.desktop.in', '[Desktop Entry]\nName=Web Browser\nKeywords=web;\nNoDisplay=true\n')
        put('xfce4-settings', 'xfsettingsd/xfsettingsd.desktop.in', '[Desktop Entry]\nName=Settings Daemon\nKeywords=settings;\n')
        with patch.object(navigation, 'OUT', root), contextlib.redirect_stdout(io.StringIO()):
            navigation.extract()
        return root

    def test_shared_keyword_unions_positives_without_negative_grades(self):
        root = self.fixture_extract()
        pairs = json.loads((root/'keyword-pairs.json').read_text())
        groups = json.loads((root/'query-groups.json').read_text())
        shared = next(g for g in groups if g['lineage_id']=='gnome:gnome-control-center' and g['query_normalized_group']=='screen')
        self.assertEqual(len(shared['positive_destination_ids']), 2)
        self.assertEqual(shared['other_destinations'], 'unjudged')
        self.assertEqual(len([p for p in pairs if p['query']=='Screen']), 1)
        for pair in pairs:
            self.assertNotIn('relevance', pair)
            self.assertNotIn('grade', pair)
            self.assertEqual(pair['positive_status'], 'inherited-author-keyword')
            self.assertTrue(pair['negative_status'].startswith('unjudged'))
            self.assertIn('/blob/', pair['source_url'])
            self.assertIn('#L', pair['source_url'])
        self.assertNotIn('heure', [p['query'] for p in pairs])
        self.assertNotIn('écran', [p['query'] for p in pairs])

    def test_roles_visibility_and_original_license_evidence_survive(self):
        root = self.fixture_extract()
        panels = json.loads((root/'panels.json').read_text())
        by_label = {p['label']: p for p in panels}
        self.assertEqual(by_label['Display']['entry_role'], 'settings-panel')
        self.assertEqual(by_label['Display']['desktop_flags']['NoDisplay'], 'true')
        self.assertEqual(by_label['Web Browser']['entry_role'], 'preferred-application-launcher')
        self.assertEqual(by_label['Settings Daemon']['entry_role'], 'background-service')
        self.assertEqual(by_label['Clock']['entry_role'], 'settings-module')
        self.assertEqual(by_label['Clock']['description'], 'Set the time')
        self.assertEqual(by_label['Keyboard']['license']['fileSpdx'], ['GPL-3.0-or-later'])
        self.assertEqual(by_label['Clock']['license']['fileSpdx'], ['LGPL-2.1-or-later'])
        self.assertIn('raw/gnome-control-center/COPYING', by_label['Display']['license']['evidenceFiles'])
        self.assertEqual(by_label['Display']['license']['fileSpdx'], [])
        self.assertIn('no file-SPDX', by_label['Display']['license']['status'])
        self.assertEqual((root/'raw/gnome-control-center/COPYING').read_text(), 'Test upstream GPL license evidence; preserved verbatim.\n')

    def test_reproduction_is_byte_identical_without_scoring(self):
        root = self.fixture_extract()
        names = ['panels.json', 'keyword-pairs.json', 'query-groups.json', 'manifest.json']
        before = {name: (root/name).read_bytes() for name in names}
        with patch.object(navigation, 'OUT', root), contextlib.redirect_stdout(io.StringIO()):
            navigation.extract()
        self.assertTrue(all((root/name).read_bytes()==data for name,data in before.items()))


if __name__ == '__main__':
    unittest.main()
