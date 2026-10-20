# Produktinfos für das Info-Fenster der Demo

Jedes Produkt hat eine eigene Datei **`<Artikelnummer>/produktinfo.txt`**
(Texteditor, UTF-8). Pro Zeile steht ein Feld, Kommentare beginnen mit `#`:

    Bezeichnung: phyCORE-i.MX 8M Plus
    Vergleich 3: 2,3 TOPS NPU für KI- und Bildverarbeitungs-Projekte

Leere Felder fallen im Fenster weg. Die Nummern [1]–[14] in den Kommentaren
zeigen, wo das Feld erscheint, siehe **`Info-Fenster_Aufbau.pdf`**:

- Seite 1: Aufbau
- Seite 2: Ansicht auf dem Monitor
- Seite 3: Vergleichsfenster (zwei Produkte angehoben)
- ab Seite 4: je Produkt die Datei links, das Fenster rechts

Muster mit allen Feldern: `PCM-070-0F3243I.A3/produktinfo.txt`.

## Für das Marketing

Am einfachsten über Word: **`Produktinfos.docx`** (eine Seite pro Produkt,
nur die Spalte „Inhalt“ ändern). Zurück in die Dateien:

    uv run tools/produktinfo_word.py import Produktinfos.docx

Neu erzeugen (nach Änderungen an den Dateien):

    uv run tools/produktinfo_word.py export

Oder direkt in den Textdateien:

- Datei mit einem normalen Texteditor öffnen (Windows-Editor, TextEdit,
  VS Code), **nicht** mit Word. Speichern als UTF-8 (Standard).
- Nur den Text **hinter** dem Doppelpunkt ändern. Der Feldname vorne bleibt.
  Groß/klein beim Feldnamen ist egal.
- Lange Texte dürfen umbrechen, auch mit Enter. Eine Folgezeile ohne
  `Feld:` gehört zum Feld darüber.
- Highlights: `Vergleich 1:` bis `Vergleich 4:` und `Alleinstellungsmerkmal 1:`
  bis `Alleinstellungsmerkmal 4:`. Der Platz zählt, leere Plätze bleiben weg.
- Zeilen mit `#` sind Hinweise und werden nicht angezeigt. Die `Bemerkung:`
  wird ebenfalls nicht angezeigt; dort steht, woher die Angaben kommen und
  was noch offen ist.
- Ob alles passt, zeigt das PDF. `uv run tools/render_panel.py` meldet
  außerdem, was nicht ins Fenster passt (z. B. „nur 7 von 8 Highlights
  passen“ oder „Kerne zu breit“).
- Die ersten Inhalte (Stand 20.10.2026) wurden von phytec.de übernommen
  und gekürzt -- bitte gegenlesen.

| Nr. | Feld | Wo |
|---|---|---|
| 1 | Bezeichnung | Titel, Kopfbalken |
| 2 | Kategorie | graue Zeile über dem Titel |
| 3 | Neu | orangefarbenes „NEU“ (ja/nein) |
| 4 | (Artikelnummer) | aus dem Ordnernamen |
| 5 | Kurzbeschreibung | max. 3 Zeilen, ≈ 270 Zeichen |
| 6–9 | Kerne, Größe in mm, Temperatur, Verfügbar bis | Kennzahlen-Zeile |
| 10 | Vergleich 1–4 | Highlights 1–4, ≈ 55 Zeichen; im Vergleichsfenster Zeile für Zeile nebeneinander, also bei allen Produkten dasselbe Thema auf demselben Platz (z. B. 1 CPU, 2 KI, 3 Speicher, 4 Schnittstellen) |
| 11 | Alleinstellungsmerkmal 1–4 | Highlights 5–8: was nur dieses Produkt kann |
| 12 | Diagramm-Datei | Bild „Im Vergleich“, Datei im Produktordner |
| 13 | Link Produktseite | QR-Code „Zum Produkt“ |
| 14 | Link Kit, Kit-Bezeichnung | QR-Code des Kits, Aufschrift des Knopfs |
| – | Aussprache (Sprachsuche), Bemerkung | nicht angezeigt |

Vorschau ohne Hardware (PDF neu erzeugen):

    uv run tools/render_panel.py

Auf das Board bringen:

    uv run tools/product_info.py models/products_v6_n
    tools/deploy.sh <board-ip> models/products_v6_n

Für ein neues Produkt:

1. in `products.csv` eintragen, fotografieren und trainieren;
2. `uv run tools/product_info.py --vorlagen` legt die leere `produktinfo.txt` an.

Die frühere Sammeltabelle liegt in `_verworfen/produktinfos.csv`. Sie wird nur
noch für Produkte ohne `produktinfo.txt` gelesen.
