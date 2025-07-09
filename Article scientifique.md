# Article scientifique récent

## Article traiter : [A Survey of techniques for Automatic Code Generation from User Interface Designs with Various Fidelities](https://drive.google.com/file/d/1mXxmnlO0iUQrIrQAN1aftrmwGpQv7-Eo/view?usp=drive_link)

### Problématique

Créer des interfaces graphiques (GUI) est un processus long, manuel et coûteux. Le processus de transformation des maquettes en code est fastidieux, souvent répétitif . Cela empêche les développeurs de se concentrer sur la logique métier, augmente les coûts et rend difficile la portabilité multi-plateformes.

---

### Approche proposée

Le document est une **revue systématique**   des techniques existantes pour générer automatiquement du code à partir de conceptions UI sous forme d’images .

Il explore :

- Des méthodes utilisant la **vision par ordinateur** .
- Des approches basées sur **l’apprentissage profond**, en particulier les CNN et les architectures encoder-decoder.
- Des approches hybrides combinant détection classique et classification par CNN.

L’objectif est de comprendre comment ces techniques extraient les éléments UI, établissent leur hiérarchie et génèrent le code source pour diverses plateformes.

---

### Technologies utilisées

Les techniques recensées utilisent principalement :

- **Réseaux de neurones convolutifs (CNN)** pour la détection et classification des éléments UI.
- **Réseaux récurrents (LSTM / RNN)** pour générer du code à partir de représentations vectorielles des interfaces.
- **Modèles de détection d’objets** comme YOLO, RetinaNet, SSD, Faster R-CNN.
- Techniques classiques de **vision par ordinateur** pour extraire les blocs UI.
- **OCR**  pour les textes manuscrits ou imprimés dans les maquettes.

Certains systèmes génèrent du code HTML/CSS/Bootstrap, d’autres des structures XML pour Android ou des DSL spécifiques.

---

### Perspectives de recherche

Le document suggère plusieurs pistes :

- Développer des **datasets standards et de haute qualité** pour comparer les approches (actuellement, chaque étude utilise son propre jeu de données).
- Définir des **cadres d’évaluation uniformes**, car les métriques varient.
- Étendre la capacité des modèles à détecter et comprendre une **plus grande diversité de composants UI** (beaucoup de travaux ne couvrent que 4 à 5 types d’éléments).
- Améliorer la reconnaissance des détails stylistiques (couleurs, polices) et la génération d’arbres hiérarchiques complexes.
- Traiter les variations comme les **images bruitées, perspectives biaisées ou fonds complexes**.
- Réduire la dépendance aux DSL rigides pour favoriser la flexibilité des composants et styles.