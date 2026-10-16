# Produktbilder zum Anlernen

Ein Ordner pro Produkt, Ordnername = Artikelnummer:

    dataset/products/
      products.csv            optional: Artikelnummer;Bezeichnung (Anzeige im Livebild)
      PCM-070-xxxx/           Bilder genau dieses Produkts
        info.tsv              Optionstabelle, tab-getrennt, 1:1 aus dem System kopiert
        summary.txt           optional: 1-3 Zeilen eigener Anzeigetext (hat Vorrang)
        img_0001.jpg
        ...
      PCM-xxx-yyyy/
        ...

Regeln, damit die Bilder automatisch beschriftet werden können:

* Pro Bild nur Exemplare **eines** Produkts (das aus dem Ordnernamen).
  Mehrere Exemplare desselben Produkts in einem Bild sind in Ordnung.
* Heller, möglichst gleichmäßiger Untergrund; das Produkt ganz im Bild.
* 20–50 Bilder pro Produkt: verschiedene Positionen, Drehungen, Abstände,
  Beleuchtung; Vorder- und Rückseite, wenn beide vorkommen können.
* Am besten mit der VM-017 aus dem späteren Aufbau (Draufsicht), Handyfotos
  gehen zusätzlich.
* Dateiformate: jpg, png. Unterordner sind erlaubt.

Bilder ohne Produkt (leerer Tisch, Hände, Kabel, fremde Platinen) gehören
nach `dataset/products/_background/` — sie verhindern Fehlalarme.
