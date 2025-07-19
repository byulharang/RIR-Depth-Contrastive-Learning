# RIR-Indoor Panorama-Contrastive-Learning
Room Impulse Response (RIR) and Indoor Panorama Contrastive without joint tasks.

**Research Period:** 2024 Summer-Autumn <br>
**Advisor:** Jung-Woo Choi (KAIST EE) <br>
**Summer Presentation:** 🔬 [Google Slide](https://docs.google.com/presentation/d/1Es7iA-b3DixLiqxxs-3j7rCpXUrvYpqKdCc9YQYZujk/edit?usp=sharing) <br>
**Data & Experiment Logs:** 🌐 [Notion](https://kiwi-primrose-e33.notion.site/16030761238f80b79c79fb76c53b808e?source=copy_link)

---

CRIP Extension Version From AV-RIR

* AV-RIR matches Late Reverberation of RIR and HorizonNet output of indoor Scene.
  * Which only uses partial information from the both modality
* We propose **General alignment between Indoor scene and RIR** by Depth & HorizonNet and full RIR
  * Mix using **depth** and **horizonNet** Outperformed in Top-1 Accuracy.

# Architecture Diagram
<img src="https://github.com/byulharang/RIR-Depth-Contrastive-Learning/blob/main/Image/Architecture.png" alt="Architecture Flow" width="600"/>

# 📄 Reference

**Ratnarajah, A., Ghosh, S., Kumar, S., Chiniya, P., & Manocha, D.**  
*AV-RIR: Audio-Visual Room Impulse Response Estimation*.  
In *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, pp. 27164–27175, 2024.

🔗 [Paper Link (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Ratnarajah_AV-RIR_Audio-Visual_Room_Impulse_Response_Estimation_CVPR_2024_paper.html)  
📄 [Download PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ratnarajah_AV-RIR_Audio-Visual_Room_Impulse_Response_Estimation_CVPR_2024_paper.pdf)

---


