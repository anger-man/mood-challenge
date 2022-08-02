---

# Was bisher geschah
 > * Daten-Aufbereitung:
 >> * DataLoader zum Einlesen von Nifti Dateien (Christoph)
 >> * Generierung zufälliger Ellipsoiden (Simon)
 >> * Generierung zufällger Quader (Christoph)
 >> * Implementierung der Störungen: Gaussian noise, random intensity, shift intensity (Christoph)
 
 > * Globales Model:
  >> * Trainiertes 3D Unet, Dice für Input der Dimension 64x64x64 liegt bei **91.85%** (Christoph)
  >> * Evaluierung des 3D Models und Export der Prädiktionen als Nifti (Christoph)
  
  
 # Nächster Schritt
 >* Implementierung des lokalen Netzwerks unter Verwendung der globalen Segmentierung (Markus)

