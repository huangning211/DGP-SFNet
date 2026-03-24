# DGP-SFNet  
**An Efficient Deformable Geometry Perception Network with Selective Fusion for Small Maritime Object Detection on Unmanned Surface Vessels**

---

## 📌 Introduction

This repository provides the official implementation of **DGP-SFNet**, a lightweight and efficient object detection network designed for **small maritime object detection** in unmanned surface vessel (USV) scenarios.

Small objects in maritime environments are challenging due to:
- limited pixel representation  
- complex backgrounds (waves, reflections, clutter)  
- diverse object scales and orientations  

To address these issues, DGP-SFNet introduces:
- **Deformable Geometry Perception (DGP)** for adaptive spatial modeling  
- **Selective Fusion (SF)** for enhanced multi-scale feature interaction  

---

## ✨ Key Contributions

- ✔ A **deformable geometry perception mechanism** to improve structural awareness of small objects  
- ✔ A **selective fusion strategy** to enhance feature representation across scales  
- ✔ A **lightweight and efficient design** suitable for real-world USV deployment  
- ✔ Strong robustness under **small-object and complex maritime environments**

---

### 🔹 models/
- Contains model architecture configuration files  
- Example: `DGP-SFNet.yaml`   

### 🔹 modules/
- Includes all **custom-designed components**

⚠️ These modules are **not included in standard Ultralytics implementations** and must be manually registered.
