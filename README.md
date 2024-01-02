# Refactoring-BCP-Project

This is a project aiming to refactor the code in [Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation (CVPR 2023)](https://github.com/DeepMed-Lab-ECNU/BCP/tree/a925e3018b23255e65a62dd34ae9ac9fc18c0bc9). The refactoring process mainly take place in the file LA_BCP_training and the files strongly related to it, hoping to make the whole project easier to read and maintain. The following images show the UML diagram before and after refactoring.

UML before:
![UML_before](https://github.com/yinghua8/Refactoring-BCP-Project/assets/71891722/63525f68-865a-4e0e-a542-26ad10630b04)

UML after:
![UML_after](https://github.com/yinghua8/Refactoring-BCP-Project/assets/71891722/535dae46-3bc5-440f-b2bf-387197023e37)


Like those who do refactoring all know, it is never enough to refactor. There are still many details that can be improved in my refactoring version; please feel free to give advice.

Note: Due to the hardware limitations of the local PC, I first modify the code in use of the CPU to prevent CUDA from running out of memory.
