# UECCT
This is the repository of a simulation program for the article: "U-shaped Error Correction Code" accepted to IEEE Transactions on Cognitive Communication and Networking (IEEE TCCN). Note that this paper is still undergoing the publication process and is not yet available on the IEEE Explore system as well as Google Scholar. An official link to the final version will provided once the paper is available on the IEEE system. 

For details of the model design and simulation results, please take a look at the PDF [here](paper/UECCT_TCCN_Accepted_ver.pdf). 

The proof of paper acceptance noticed by IEEE TCCN can be viewed [here](paper/TCCN_AcceptanceNotice.pdf).

## Abstract
In this work, we introduce two variants of the U-shaped error correction code transformer (U-ECCT) in combination with weight-sharing to improve the decoding performance of the error correction code transformer (ECCT) for moderate-length linear codes. The proposed models are inspired by the well-known U-Net architecture to leverage residual information for faster error estimation based on the syndrome-based reliability decoding principle. As an effort to further improve the general decoding performance of the U-ECCT, we propose the variational U-ECCT (VU-ECCT), in which the process of learning the shortcut connections is treated as a generative problem, forming a variational autoencoder (VAE) that exists intertwined with the existing U-ECCT model. This design allows the extraction of mutual information between the different levels of the U-shaped architecture, thus enhancing the performance of large syndrome sequences for low-rate codes. Additionally, to further reduce the model size, a new weight-sharing strategy, called mirror-sharing, is proposed to compress the model size as well as complement the mechanism of the proposed U-shaped architecture. In experiments, it has been demonstrated that our proposed models achieve significantly better performance than baseline conventional algorithms and other learning-based models.


