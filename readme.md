---

# Was bisher geschah
 > * Daten-Aufbereitung:
 >> * DataLoader zum Einlesen von Nifti Dateien (Christoph)
 >> * Generierung zufälliger Ellipsoiden (Simon)
 >> * Generierung zufällger Quader (Christoph)
 >> * Implementierung der Störungen: Gaussian noise, random intensity, shift intensity, Sobel filter, scale intensity (Christoph)
 >> * Optimierung des Data-Loader (CC size, CC amount, perturbation probability, ...) (Christoph)
 
 > * Globales Model:
  >> * Trainiertes 3D Unet, Dice für Input der Dimension 64x64x64 liegt bei **91.85%** (Christoph)
  >> * Evaluierung des 3D Models und Export der Prädiktionen als Nifti (Christoph)
  >> * **TBD: Dice Loss ist keine vernünftige Fehlerfunktion bei Vorkommen von ungestörten Daten**
  
 > * Docker Container:
  >> * Einarbeiten in Docker und in die für die Challenge notwendige Submission Architektur
  >> * Erstellen eines funktionierenden Docker Images für globale Prädiktionen, funktioniert auch mit dem MOOD test-skript (Christoph)
  
# Nächster Schritt
 > * Implementierung des lokalen Netzwerks unter Verwendung der globalen Segmentierung (Markus)

