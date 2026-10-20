# Produktinfos für die Demo-Anzeige

`produktinfos.csv` (Excel: Datei öffnen, Trennzeichen Semikolon, UTF-8) –
eine Zeile pro Produkt, Artikelnummer wie in `products.csv`. Leere Felder
bleiben einfach leer.

| Spalte | Wofür | Beispiel |
|---|---|---|
| Artikelnummer | Schlüssel, wie in products.csv | PCM-070-0F3243I.A3 |
| Bezeichnung | Titel im Info-Panel | phyCORE-i.MX8M Plus |
| Kategorie | graue Zeile über dem Titel, Brotkrumen | System on Module |
| Neu (ja/nein) | orangefarbenes „NEU“ | ja |
| Kerne / Größe in mm / Temperatur / Verfügbar bis | Kennzahlen-Zeile unter dem Titel | 4 × A53 + M7 / 40 × 37 × 6 / -40 °C / +85 °C / 2035 |
| Highlight 1–8 | Liste mit Häkchen, je eine kurze Zeile (≤ 55 Zeichen) | NPU mit 2,3 TOPS für KI-Anwendungen |
| Link Mehr Infos | QR-Code unten rechts | https://www.phytec.de/... |
| Diagramm-Datei | Bild im Produktordner, links neben dem QR-Code | chart.png |
| Aussprache (Sprachsuche) | wie man den Namen spricht (für spätere Sprachsuche) | i m x acht m plus |
| Bemerkung | nur für euch, wird nicht angezeigt | |

Ohne Highlights zeigt das Panel die Optionen aus `info.tsv` des
Produktordners. Neue Produkte: zuerst in `products.csv` eintragen
(fotografieren und trainieren), dann hier eine Zeile ergänzen.
Änderungen kommen mit `tools/product_info.py <model>` bzw. dem nächsten
Modell-Build auf das Board.
