# Produktinfos für das Info-Fenster der Demo

Jedes Produkt hat eine eigene Datei **`<Artikelnummer>/produktinfo.txt`**
(Texteditor, UTF-8). Pro Zeile steht ein Feld, Kommentare beginnen mit `#`:

    Bezeichnung: phyCORE-i.MX 8M Plus
    Highlight: 2,3 TOPS NPU für KI- und Bildverarbeitungs-Projekte

Leere Felder fallen im Fenster weg. Die Nummern [1]–[13] in den Kommentaren
zeigen, wo das Feld erscheint, siehe **`Info-Fenster_Aufbau.pdf`**:

- Seite 1: Aufbau
- Seite 2: Ansicht auf dem Monitor
- ab Seite 3: je Produkt die Datei links, das Fenster rechts

Muster mit allen Feldern: `PCM-070-0F3243I.A3/produktinfo.txt`.

| Nr. | Feld | Wo |
|---|---|---|
| 1 | Bezeichnung | Titel, Kopfbalken |
| 2 | Kategorie | graue Zeile über dem Titel |
| 3 | Neu | orangefarbenes „NEU“ (ja/nein) |
| 4 | (Artikelnummer) | aus dem Ordnernamen |
| 5 | Kurzbeschreibung | max. 3 Zeilen, ≈ 270 Zeichen |
| 6–9 | Kerne, Größe in mm, Temperatur, Verfügbar bis | Kennzahlen-Zeile |
| 10 | Highlight (bis zu 8 Zeilen) | Liste mit Häkchen, ≈ 55 Zeichen |
| 11 | Diagramm-Datei | Bild „Im Vergleich“, Datei im Produktordner |
| 12 | Link Produktseite | QR-Code „Zum Produkt“ |
| 13 | Link Kit, Kit-Bezeichnung | QR-Code des Kits, Aufschrift des Knopfs |
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
